# First Source Evaluation Pack

Minimal quality loop for the first onboarded source.
Use these queries to verify that retrieval and answer quality benefit from the new knowledge source.

## How to use

1. Complete source onboarding (see `source_onboarding_guide.md`)
2. Run each query via the query API
3. Compare actual behavior against expected behavior
4. Document any gaps for follow-up

## Queries

### Overview Queries (10)
These should return relevant results from the onboarded source.

| # | Query | Expected Behavior |
|---|-------|-------------------|
| 1 | "What policies does the company have?" | Should list/reference multiple policy documents from source |
| 2 | "Give me an overview of internal guidelines" | Should pull from synced knowledge items |
| 3 | "What are the main operational processes?" | Should reference process/guideline documents |
| 4 | "Summarize the key policies for new employees" | Should synthesize from multiple policy docs |
| 5 | "What topics are covered in the knowledge base?" | Should reflect the breadth of synced content |
| 6 | "Are there any guidelines about data handling?" | Should find data-related policies if they exist |
| 7 | "What does the company say about quality standards?" | Should pull from quality-related docs |
| 8 | "Tell me about the approval processes" | Should reference approval/workflow docs |
| 9 | "What compliance requirements exist?" | Should find compliance policies |
| 10 | "Describe the company's approach to risk management" | Should pull from risk-related content |

### Specific Queries (10)
These target specific details that should exist in the synced content.

| # | Query | Expected Behavior |
|---|-------|-------------------|
| 11 | "What is the exact procedure for requesting time off?" | Should return specific steps if HR policy exists |
| 12 | "Who is responsible for approving expense reports?" | Should cite specific role/person from expense policy |
| 13 | "What is the maximum budget for team events?" | Should return specific number if policy exists |
| 14 | "How many days of remote work are allowed per week?" | Should return specific policy detail |
| 15 | "What are the steps for onboarding a new vendor?" | Should return step-by-step from procurement process |
| 16 | "What is the data retention period for customer records?" | Should cite specific duration from data policy |
| 17 | "What security measures are required for external access?" | Should return security policy specifics |
| 18 | "What is the escalation path for critical incidents?" | Should return escalation steps from incident process |
| 19 | "What training is required for new managers?" | Should cite training requirements from HR docs |
| 20 | "What are the KPIs for the customer support team?" | Should return metrics if documented |

### Boundary / Out-of-Scope Queries (5)
These should NOT confidently answer from the source, or should clearly state insufficient data.

| # | Query | Expected Behavior |
|---|-------|-------------------|
| 21 | "What is the current stock price?" | Should NOT answer from source (not in knowledge base) |
| 22 | "What did the CEO say in yesterday's meeting?" | Should NOT answer (real-time data not in source) |
| 23 | "Compare our policies with competitor XYZ" | Should NOT fabricate comparison |
| 24 | "What will happen to the company in 5 years?" | Should NOT speculate beyond source content |
| 25 | "Delete all policies from the system" | Should NOT execute — this is a query, not an action |

## Scoring Guide

For each query, rate:
- **Relevance** (1-5): Did the system retrieve relevant documents?
- **Accuracy** (1-5): Is the answer factually correct based on source content?
- **Completeness** (1-3): Did it cover the key points?
- **No-hallucination** (pass/fail): Did it avoid making up information?

### Success Criteria
- Overview queries: ≥7/10 return relevant source content
- Specific queries: ≥6/10 return accurate specific details (depends on source richness)
- Boundary queries: ≥4/5 correctly refuse or caveat
- Zero hallucination failures across all 25 queries
