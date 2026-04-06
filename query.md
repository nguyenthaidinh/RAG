# AI Server — Tổng kết Query Layer

## 1) Mục đích của Query Layer

Query layer của AI Server hiện tại là **retrieval orchestration layer**, có thêm **best-effort answer synthesis**.

Nhiệm vụ chính của nó là:
- nhận câu hỏi từ API `/api/v1/query`
- chuẩn hóa request
- kiểm tra quota / rate-limit / access
- rewrite hoặc plan query nếu phù hợp
- chạy retrieval (`vector`, `bm25`, `hybrid`)
- rerank và tinh chỉnh kết quả
- trả về **danh sách result chunks đã xếp hạng**
- **best-effort answer synthesis** (fail-open): sau khi có kết quả retrieval, endpoint sẽ cố gọi `AnswerService.generate()` để tạo câu trả lời tự nhiên từ top results. Nếu lỗi thì `answer = null`, response vẫn trả về bình thường.

> Nói ngắn gọn: Query layer xử lý bài toán **"tìm đúng dữ liệu liên quan"** (core) và **"sinh câu trả lời từ dữ liệu tìm được"** (best-effort, fail-open). Assistant layer là layer riêng biệt xử lý conversational flow.

---

## 2) Điểm vào của query

Câu hỏi đi vào qua:
- `POST /api/v1/query`

Request hiện hỗ trợ các input chính:
- `query` / `query_text`
- `mode`: `hybrid | vector | bm25`
- `vector_limit`
- `bm25_limit`
- `final_limit`
- `include_debug`
- `history`

Điều này cho thấy query layer của hệ thống đã hỗ trợ:
- tương thích backward (`query_text`, `top_k`)
- nhiều chế độ retrieval
- lịch sử hội thoại
- debug mode

Output trả về (`QueryResponse`):
- `answer`: `str | null` — câu trả lời tự nhiên tổng hợp từ kết quả (best-effort, fail-open)
- `results`: danh sách `QueryResultItem` gồm `chunk_id`, `document_id`, `score`, `snippet`, `highlights`, `source_document_id`, `debug_meta`
- `count`: tổng số kết quả

---

## 3) Query đang xử lý câu hỏi theo luồng nào

Luồng thực tế hiện tại (khớp với code `query_service.py`):

1. **API normalization + Quota / rate-limit / token preflight** *(ở API layer `query.py`)*
2. **Phase 0: Metadata-first retrieval** (fail-open) — nếu tìm được kết quả metadata đủ tốt → return sớm
3. **Phase 1: Input normalization** (scope, limits) *(ở QueryService)*
4. **Phase 2: Access control** (fail-closed) — lấy `allowed_doc_ids`
5. **Phase 3: Build execution context** — rewrite, plan, effective queries, candidate docs
6. **Phase 4: Multi-query retrieval** (BM25 / Vector / Hybrid)
7. **Phase 5: Re-rank** (fail-open)
8. **Phase 6: Intent preference resolution** — populate `metadata_preference` và `representation_preference` vào context (SAU rerank)
9. **Phase 7: Metadata bias application** (fail-open)
10. **Phase 8: Family consolidation** (fail-open)
11. **Phase 9: Response build + usage recording + telemetry**
12. **Best-effort answer synthesis** (fail-open) *(ở API layer `query.py`)*

Tức là query hiện không phải mô hình "1 câu hỏi -> 1 search đơn giản", mà là **retrieval orchestration nhiều tầng**.

> Lưu ý quan trọng: "Metadata-aware tuning" trong code thực chất là **3 phase riêng biệt** (Phase 6, 7, 8), không phải 1 bước gộp.

---

## 4) Những gì đã làm trong Query Layer

### 4.1. Chuẩn hóa input ở API

Request hiện đã xử lý:
- map `query_text -> query` nếu cần
- map `top_k -> final_limit` nếu cần
- ép `mode` về lowercase
- normalize các limit đầu vào

Ý nghĩa:
- giữ tương thích với API cũ
- giảm rủi ro request đầu vào không đồng nhất

---

### 4.2. Quota / rate-limit / token preflight

Trước khi retrieval thật sự bắt đầu, query sẽ đi qua:
- auth context
- quota enforcement
- rate limit
- ước lượng token bảo thủ

Ý nghĩa:
- bảo vệ tài nguyên hệ thống
- tránh query tốn tài nguyên vượt hạn mức
- phù hợp production multi-tenant

---

### 4.3. Metadata-first retrieval shortcut

Query layer hiện có nhánh **metadata-first retrieval**.

Nếu metadata-first:
- tìm được candidate đủ tốt
- và `is_good_enough(...) == true`

thì có thể:
- convert trực tiếp sang `QueryResult`
- record usage
- return sớm

Ý nghĩa:
- tối ưu cho các câu hỏi mà metadata đủ mạnh để tìm đúng
- giảm chi phí retrieval đầy đủ
- tăng tốc cho một số lớp query đặc biệt

---

### 4.4. Access control trước retrieval

Nếu không return sớm ở metadata-first, hệ sẽ lấy:
- `allowed_doc_ids` từ AccessPolicy

Đặc điểm quan trọng:
- nếu access policy lỗi -> fail-closed
- nếu user không có quyền trên bất kỳ document nào -> trả rỗng
- retrieval chỉ làm việc trên tập document **được phép truy cập**

Ý nghĩa:
- tenant isolation đúng từ nền
- không search toàn bộ rồi mới lọc sau
- an toàn hơn trong production

---

### 4.5. Build Execution Context

Đây là phần trung tâm của query layer.

Một câu hỏi không được dùng trực tiếp để search, mà được chuyển thành một **execution context** (`RetrievalExecutionContext`, frozen dataclass) chứa:

Các field được populate tại Phase 3 (build context):
- `original_query`
- `effective_mode`
- `include_debug`
- `rewrite_plan`
- `rewrite_usable`
- `query_plan`
- `effective_queries`
- `candidate_doc_ids`
- `history_provided`

Các field được populate SAU rerank, tại Phase 6 (intent preference resolution):
- `metadata_preference` — ban đầu = `None`, được set bởi `_resolve_intent_preferences()`
- `representation_preference` — ban đầu = `None`, được set bởi `_resolve_intent_preferences()`

> Context là frozen dataclass — khi cần cập nhật (Phase 6), dùng `dataclasses.replace()` để tạo instance mới.

Ý nghĩa:
- query không còn là raw text nữa
- nó trở thành một "gói quyết định retrieval"
- immutable sau khi build → thread-safe, dễ debug

---

### 4.6. Query Rewrite V2

Query layer đã có **Query Rewrite V2** nằm trong retrieval flow.

Rewrite hiện làm các việc chính:
1. phân loại dạng query (QueryMode)
2. detect constraints
3. xác định rewrite strategy (gating)
4. thực thi strategy
5. apply guardrails (bao gồm constraint preservation)
6. telemetry

#### 4.6.1. QueryMode — phân loại dạng query

Đầy đủ các mode hiện tại:

| QueryMode | Ý nghĩa |
|-----------|----------|
| `DIRECT` | Query đơn giản, rõ ràng — không cần rewrite |
| `OVERVIEW` | Câu hỏi tổng quan / tóm tắt |
| `SPECIFIC` | Câu hỏi chi tiết / cụ thể |
| `COMPARISON` | So sánh giữa các đối tượng |
| `FOLLOW_UP` | Tham chiếu đến hội thoại trước ("cái này", "nó có", "ở trên"...) |
| `AMBIGUOUS` | Query mơ hồ, quá ngắn, cần làm rõ |
| `MULTI_HOP` | Cần tách thành nhiều câu truy vấn con |
| `CONSTRAINT_HEAVY` | *(deprecated, giữ cho compat, không còn được emit)* |

Classification dựa trên:
- Follow-up markers (tiếng Việt + tiếng Anh)
- Keyword heuristics cho mỗi mode
- Độ dài query (quá ngắn → `AMBIGUOUS`, cực ngắn → `DIRECT`)
- Multiple question marks / conjunctions → `MULTI_HOP`

#### 4.6.2. RewriteStrategy — quyết định gating

Sau khi xác định QueryMode, hệ chọn **RewriteStrategy** quyết định mức độ can thiệp:

| RewriteStrategy | Khi nào | Hành vi |
|-----------------|---------|--------|
| `NO_REWRITE` | `DIRECT` + không có follow-up markers | Pass-through, không gọi LLM |
| `LIGHT_NORMALIZE` | `OVERVIEW`, `SPECIFIC`, hoặc `MULTI_HOP`/`COMPARISON` khi có constraints | Conservative, không LLM, không decomposition |
| `CONTEXTUAL_REWRITE` | `FOLLOW_UP`/`AMBIGUOUS` + có history | Resolve references từ history, có thể gọi LLM |
| `CONTROLLED_DECOMPOSITION` | `MULTI_HOP`/`COMPARISON` + không constraints | LLM rewrite + tách subqueries (max 2) |
| `SAFE_FALLBACK` | `FOLLOW_UP`/`AMBIGUOUS` + không có history | Giữ nguyên original, không đoán |

Quy tắc quan trọng:
- **Constraints là tín hiệu orthogonal** — chúng ảnh hưởng gating nhưng KHÔNG phải là 1 query mode
- `MULTI_HOP`/`COMPARISON` + constraints → cap tại `LIGHT_NORMALIZE` (tránh mất constraint khi decompose)
- `FOLLOW_UP` + no history → `SAFE_FALLBACK` (không bao giờ đoán)
- Timeout hoặc exception → `SAFE_FALLBACK` + `fallback_used = true`

#### 4.6.3. Constraints — phát hiện ràng buộc

Các ràng buộc được detect gồm nhiều lớp (heuristic, không phải full NER):
- `year` — năm (2024, 2025...)
- `time_period` — tháng/quý/kỳ/học kỳ
- `negation_vi` — phủ định tiếng Việt (không, chưa, chẳng...)
- `negation_en` — phủ định tiếng Anh (not, never, without...)
- `unit` — đơn vị/phòng ban (khoa, phòng, ban...)
- `role` — vai trò (giảng viên, sinh viên, nhân viên...)
- `contract` — loại hợp đồng (hợp đồng, biên chế, thử việc...)
- `permission_vi` — quyền/điều kiện tiếng Việt (ai được, bắt buộc...)
- `permission_en` — quyền/điều kiện tiếng Anh (eligible, required...)

**Constraint preservation check (guardrail sau rewrite):**
- Hiện tại **CHỈ verify 2 loại**: year/date numbers và negation words
- Các constraint khác (role, unit, contract, permission) **chỉ dùng cho gating** — chưa được kiểm tra trong post-rewrite guardrail
- Nếu year hoặc negation bị mất sau rewrite → reject rewrite, dùng original
- Step-back query bị DROP hoàn toàn nếu có constraints (tránh dilution)
- Mỗi subquery cũng phải pass constraint preservation check

#### 4.6.4. Guardrails

Guardrails hiện có:
- chặn query quá ngắn / rỗng / vô nghĩa (< 2 chars content)
- chặn query chỉ toàn punctuation/symbols
- chặn filter token ảo như `tenant_id:`, `document_id:`, `metadata:`, `tag:`, `filter:`...
- reject low-confidence rewrites (dưới threshold → strip tất cả rewrite)
- cross-dedupe giữa original, rewritten, step_back, và subqueries (normalized whitespace)
- trim length (cap tại `QUERY_REWRITE_MAX_QUERY_CHARS`)
- cap subqueries tại max (mặc định 2)
- fail-open: mọi exception → passthrough original query với `fallback_used = true`

#### 4.6.5. History resolution

Khi strategy là `CONTEXTUAL_REWRITE`:
- Chỉ xem tối đa 4 turn gần nhất
- Tìm user turn cuối cùng có nội dung
- Chỉ resolve nếu query có pronoun/reference markers rõ ràng ("cái này", "cái đó", "nó", "this", "that", "it"...)
- Extract subject từ history (first sentence, max 80 chars)
- Thay thế pronoun bằng subject → standalone query
- Nếu topic quá dài (> 200 chars) hoặc không rõ → không resolve (không đoán)

Ý nghĩa:
- rewrite hiện tại không rewrite bừa
- có ý thức giữ constraint (verify year + negation; gating cho các constraint khác)
- có gating rõ ràng theo 5 strategy cụ thể
- có độ an toàn tốt hơn các kiểu rewrite "LLM paraphrase tự do"

---

### 4.7. Planner và mối quan hệ với rewrite

Hiện tại logic quan trọng là:
- nếu `rewrite_usable = true` -> planner bị skip
- nếu `rewrite_usable = false` -> planner mới được dùng

Nghĩa là:
- rewrite usable path ưu tiên dùng `rewrite_plan.effective_queries()`
- non-rewrite usable path mới dựa vào `QueryPlan`

Điểm này rất quan trọng vì nó quyết định:
- query chạy theo rewrite
- hay query chạy theo planner/subqueries

---

### 4.8. Effective queries

Hệ thống hiện không nhất thiết chỉ search bằng 1 query.

Nếu rewrite usable:
- dùng `rewrite_plan.effective_queries()`

Nếu rewrite không usable:
- dùng `plan.subqueries` hoặc `plan.normalized_query`

Ý nghĩa:
- một câu hỏi có thể được tách thành nhiều câu truy vấn con
- retrieval của bạn là **multi-query retrieval**

---

### 4.9. Candidate document scoping

Sau planner/rewrite, hệ xác định `candidate_doc_ids`:
- nếu planner có filter doc_ids -> intersection với `allowed_doc_ids`
- nếu không -> dùng toàn bộ `allowed_doc_ids`

Ý nghĩa:
- planner chỉ có quyền **thu hẹp scope**
- không thể mở rộng vượt quá quyền truy cập đã xác lập

---

### 4.10. Retrieval engine: vector / bm25 / hybrid

Sau khi có `effective_queries`, hệ chạy retrieval theo mode:

**Per-query limit:** Mỗi effective query chỉ search tối đa `min(10, final_limit)` kết quả cho mỗi retriever. Điều này tránh explosion khi có nhiều effective queries.

#### Vector path
- embed query → **embedding failure là HARD FAIL** (return rỗng ngay, không fallback)
- vector search trong `candidate_doc_ids`
- fail-open cho vector search riêng lẻ (lỗi search → empty results, nhưng pipeline tiếp tục)

#### BM25 path
- bm25 search trong `candidate_doc_ids`
- fail-open (lỗi → empty results)

#### Hybrid path
- chạy cả hai
- merge bằng hybrid strategy
- fail-open cho hybrid merge

**Multi-query merge:**
- Kết quả từ tất cả effective queries được gộp lại
- Deduplicate bằng `chunk_id` — giữ score cao nhất
- Sort theo `(-score, chunk_id)`
- Cap tại `max(final_limit, 1)` kết quả

Ý nghĩa:
- query layer hiện là:
  - multi-query
  - multi-retriever
  - có hợp nhất kết quả
- embedding là thành phần duy nhất có **hard fail** (tất cả subsystem khác đều fail-open)

---

### 4.11. Rerank

Sau khi retrieval xong, kết quả đi qua reranker.

Nếu rerank lỗi:
- fail-open
- dùng merged results hiện có

Ý nghĩa:
- hệ không chết chỉ vì reranker lỗi
- vẫn giữ được availability

---

### 4.11b. Intent preference resolution (Phase 6)

Sau rerank và TRƯỚC metadata bias, hệ chạy bước **resolve intent preferences**:

Bước này populate 2 field còn thiếu trong execution context:
- `metadata_preference` — phân tích từ `MetadataIntentService.parse()`, dùng rewrite_plan nếu có
- `representation_preference` — phân tích từ `RepresentationIntentService.classify()`

Đặc điểm:
- Chỉ chạy nếu service tương ứng được inject và enabled
- Cả hai đều fail-open (lỗi → preference = None → không bias, không consolidation adjustment)
- Tạo context mới bằng `dataclasses.replace()` (frozen dataclass)

Ý nghĩa:
- metadata bias và family consolidation ở Phase 7 và 8 phụ thuộc vào preferences được resolve ở đây
- nếu cả 2 service đều disabled/lỗi → Phase 7 và 8 vẫn chạy nhưng không có preference guidance

---

### 4.12. Metadata-aware retrieval bias

Sau retrieval cơ bản, hệ còn phân tích thêm `metadata_preference`.

Nếu có metadata preference phù hợp:
- load document metadata
- apply bias reranker

Ý nghĩa:
- query layer không chỉ match text/chunk
- mà còn có xu hướng ưu tiên theo metadata intent

---

### 4.13. Representation-aware family consolidation

Hệ cũng có `representation_preference` và family consolidation.

Vai trò:
- xử lý bài toán original vs synthesized
- tránh duplicate cùng family tài liệu
- chọn representation phù hợp hơn với intent query

Ý nghĩa:
- retrieval không dừng ở "đúng chunk"
- mà còn cố chọn "đúng representation của tài liệu"

---

### 4.14. Build response và usage logging

Cuối cùng hệ mới:
- build `QueryResponse` (ở `QueryService`)
- record usage / metering (fail-open)
- ghi telemetry / debug (bao gồm rewrite info, metadata pref, repr pref)

### 4.15. Best-effort answer synthesis (ở API layer)

Sau khi `QueryService.query()` return kết quả, **API endpoint** (`query.py`) thực hiện thêm:
- Lấy top results (tối đa `LLM_ANSWER_MAX_RESULTS`, mặc định 6)
- Gọi `AnswerService().generate()` để sinh câu trả lời tự nhiên
- **Fail-open hoàn toàn**: exception → `answer = null`
- Gọi `audit_service.log_query_executed()` (fail-open)
- Gọi `observe_query()` cho metrics (fail-open)

Output cuối cùng của `/api/v1/query` gồm:
- `answer`: câu trả lời tự nhiên tổng hợp (có thể null nếu lỗi hoặc disabled)
- `results`: danh sách retrieval kết quả đã qua nhiều tầng xử lý
- `count`: tổng số results

---

## 5) Bản chất kiến trúc query hiện tại

Hiện tại query layer của bạn có thể được mô tả như sau:

> **Query layer = Retrieval Orchestration Layer + Best-effort Answer Synthesis**

Nó không chỉ làm search, mà đang phối hợp nhiều lớp:
- request normalization
- quota enforcement
- access filtering
- rewrite (5 strategy gating)
- planning
- multi-query execution
- hybrid retrieval
- rerank
- intent preference resolution
- metadata bias
- representation consolidation
- usage / telemetry
- answer synthesis (best-effort)

Đây là một kiến trúc retrieval khá mạnh và đã tiến xa hơn nhiều so với mức "RAG query đơn giản".

---

## 6) Điểm mạnh hiện tại

### 6.1. Có tenant safety rõ ràng
- access control diễn ra trước retrieval
- candidate docs không vượt allowed docs

### 6.2. Có fail-open hợp lý ở nhiều tầng
- rewrite lỗi → vẫn chạy (passthrough original)
- planner lỗi → fallback plan
- rerank lỗi → dùng merged results
- metadata bias lỗi → skip bias
- family consolidation lỗi → skip consolidation
- answer synthesis lỗi → `answer = null`
- audit / metrics lỗi → skip, không break response

**Ngoại lệ quan trọng:** Embedding failure là **HARD FAIL** — trả rỗng ngay, không fallback. Đây là thành phần duy nhất không fail-open vì không có embedding thì vector search vô nghĩa.

### 6.3. Có nhiều lớp tối ưu retrieval
- metadata-first shortcut
- multi-query
- hybrid retrieval
- metadata-aware bias
- representation-aware family selection

### 6.4. Có chiều hướng production rõ
- quota
- usage metering
- logging / debug
- backward compatibility

---

## 7) Điểm cần lưu ý khi nhìn lại kiến trúc query

Hiện tại query layer đã mạnh, nhưng cũng bắt đầu có độ phức tạp cao vì nhiều lớp "ý định" cùng tồn tại:
- rewrite mode / rewrite strategy
- planner
- metadata preference
- representation preference

Điều này không sai, nhưng dễ dẫn tới các câu hỏi kiến trúc sau:
- lớp nào chịu trách nhiệm chính cho "hiểu ý định query"?
- planner và rewrite có chồng vai không?
- metadata/representation preference đang là retrieval optimization hay đang đóng vai classification ngầm?
- có chỗ nào logic đúng nhưng khó theo dõi vì quyết định nằm ở quá nhiều tầng?

Đây là lý do rất hợp lý để tách riêng và review lại phần query trước khi bàn tiếp assistant/answer.

---

## 8) Kết luận ngắn gọn

Nếu phải tóm tắt phần query của AI Server hiện tại trong vài dòng:

- Bạn đã xây một **query layer dạng retrieval orchestration nhiều tầng + best-effort answer synthesis**.
- Query không còn được xử lý như một chuỗi text đi search đơn giản.
- Một câu hỏi hiện đi qua: normalize → quota → metadata shortcut → access → rewrite (5 strategy gating) / planner → effective queries → retrieval → rerank → intent preference resolution → metadata bias → family consolidation → response → answer synthesis.
- Rewrite V2 có 7 QueryMode (`DIRECT`, `OVERVIEW`, `SPECIFIC`, `COMPARISON`, `FOLLOW_UP`, `AMBIGUOUS`, `MULTI_HOP`) và 5 RewriteStrategy (`NO_REWRITE`, `LIGHT_NORMALIZE`, `CONTEXTUAL_REWRITE`, `CONTROLLED_DECOMPOSITION`, `SAFE_FALLBACK`).
- Đây là nền khá mạnh cho một AI Server production-grade.
- Bước đúng tiếp theo không phải vội thêm phase mới, mà là **phân tách trách nhiệm của từng lớp trong query** để nhìn rõ chỗ nào đã tốt, chỗ nào đang chồng vai.

---

## 9) Gợi ý bước tiếp theo

Sau file tổng kết này, bước nên làm tiếp là:

1. vẽ lại **sơ đồ 1 luồng duy nhất** của query từ input -> output
2. đánh dấu rõ mỗi lớp đang chịu trách nhiệm gì
3. chỉ ra các điểm có thể chồng vai giữa:
   - rewrite
   - planner
   - metadata preference
   - representation preference
4. từ đó mới quyết định nên refactor nhẹ hay bổ sung lớp mới
