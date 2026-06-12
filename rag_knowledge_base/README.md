# Financial Advisory RAG Knowledge Base

A curated collection of authoritative financial documents for a multi-agent financial advisory system. The advisory agent queries this knowledge base to ground its recommendations in verified sources rather than parametric memory.

---

## Directory Structure

```
rag_knowledge_base/
├── README.md                          ← this file
├── download_pdfs.py                   ← script to download remaining PDFs
├── regulatory/                        ← SEC, FINRA, CFPB, FDIC
│   ├── SEC_Asset_Allocation_Diversification.txt
│   ├── SEC_Investment_Risk_Guide.txt
│   ├── SEC_Mutual_Funds_Guide.txt
│   ├── SEC_ETF_Guide.txt
│   ├── SEC_Bonds_Fixed_Income_Guide.txt
│   ├── SEC_Stocks_Guide.txt
│   └── SEC_Risk_Tolerance_Time_Horizon.txt
├── central_bank/                      ← Federal Reserve, BIS, IMF (PDFs via script)
│   ├── FINRA_Investment_Strategies_Guide.txt
│   ├── FedStLouis_Economic_Overview.txt
│   └── pdfs/                          ← populated by download_pdfs.py
│       ├── FINRA_Smart_Bond_Investing.pdf
│       ├── FINRA_Saving_Investing_Roadmap.pdf
│       ├── BIS_Quarterly_Review_2025_Sep.pdf
│       ├── BIS_Quarterly_Review_2025_Dec.pdf
│       ├── BIS_Quarterly_Review_2026_Mar.pdf
│       ├── BIS_Annual_Report_2025.pdf
│       ├── IMF_GFSR_2025_Oct.pdf
│       ├── IMF_GFSR_2026_Apr.pdf
│       ├── IMF_WEO_2025_Oct.pdf
│       ├── Fed_Monetary_Policy_Report_2025.pdf
│       └── Fed_Monetary_Policy_Report_2025_Jun.pdf
├── institutional_research/            ← Vanguard, Fidelity, JP Morgan, iShares
│   ├── Vanguard_Institutional_Overview.txt
│   ├── Fidelity_Learning_Center_Overview.txt
│   └── pdfs/                          ← populated by download_pdfs.py + manual
│       ├── Vanguard_Principles_Investing_Success.pdf  [manual download]
│       ├── Vanguard_Advisors_Alpha.pdf                [manual download]
│       ├── Vanguard_Economic_Outlook_2026.pdf         [manual download]
│       ├── JPM_Guide_to_Markets_CURRENT.pdf           [manual download]
│       ├── iShares_ETF_Investing_Guide.pdf
│       └── Schwab_Asset_Allocation_Guide.pdf          [manual download]
└── academic/                          ← arXiv q-fin papers
    └── pdfs/                          ← populated by download_pdfs.py
        ├── arXiv_Portfolio_Optimization_Survey.pdf
        ├── arXiv_Risk_Parity_Portfolios.pdf
        ├── arXiv_Drawdown_Risk_Measures.pdf
        ├── arXiv_Modern_Portfolio_Theory_Review.pdf
        ├── arXiv_Factor_Investing.pdf
        ├── arXiv_Sharpe_Ratio_Estimation.pdf
        └── arXiv_CVaR_Portfolio_Optimization.pdf
```

---

## Document Inventory

### Already Available (Text Files — Ready for Ingestion)

| File | Source | Authority | Key Topics |
|------|--------|-----------|------------|
| `regulatory/SEC_Asset_Allocation_Diversification.txt` | SEC investor.gov | ★★★★★ Regulatory | Asset allocation, diversification, rebalancing, time horizon |
| `regulatory/SEC_Investment_Risk_Guide.txt` | SEC investor.gov | ★★★★★ Regulatory | Risk types: business, volatility, inflation, interest rate, liquidity |
| `regulatory/SEC_Mutual_Funds_Guide.txt` | SEC investor.gov | ★★★★★ Regulatory | Mutual fund types, fees, index vs active, risks |
| `regulatory/SEC_ETF_Guide.txt` | SEC investor.gov | ★★★★★ Regulatory | ETF structure, tax advantages, intraday trading, risks |
| `regulatory/SEC_Bonds_Fixed_Income_Guide.txt` | SEC investor.gov | ★★★★★ Regulatory | Corporate/muni/Treasury bonds, credit risk, interest rate risk |
| `regulatory/SEC_Stocks_Guide.txt` | SEC investor.gov | ★★★★★ Regulatory | Common/preferred stock, market cap, growth/value/income categories |
| `regulatory/SEC_Risk_Tolerance_Time_Horizon.txt` | SEC investor.gov | ★★★★★ Regulatory | Risk tolerance assessment, short vs long horizon |
| `central_bank/FINRA_Investment_Strategies_Guide.txt` | FINRA | ★★★★★ Regulatory | Active/passive strategies, DCA, value/momentum investing |
| `central_bank/FedStLouis_Economic_Overview.txt` | Federal Reserve (St. Louis) | ★★★★★ Official | Current macro indicators (May 2026), Fed role, rate impact |
| `institutional_research/Vanguard_Institutional_Overview.txt` | Vanguard | ★★★★ Institutional | 4 investment principles, Advisor's Alpha, TDF philosophy |
| `institutional_research/Fidelity_Learning_Center_Overview.txt` | Fidelity | ★★★★ Institutional | ETF portfolio framework, 2026 midyear outlook |

### Requires Running `download_pdfs.py`

| File | Source | Priority | Notes |
|------|--------|----------|-------|
| `FINRA_Smart_Bond_Investing.pdf` | FINRA | High | Fixed income fundamentals |
| `FINRA_Saving_Investing_Roadmap.pdf` | FINRA | High | Savings and investment planning |
| `CFPB_Your_Money_Your_Goals.pdf` | CFPB | High | Personal finance toolkit |
| `FDIC_Money_Smart_Intro.pdf` | FDIC | Medium | Consumer financial literacy |
| `BIS_Quarterly_Review_*.pdf` (×3) | BIS | **Highest** | Global financial conditions, macro |
| `BIS_Annual_Report_2025.pdf` | BIS | **Highest** | Annual economic survey |
| `IMF_GFSR_*.pdf` (×2) | IMF | **Highest** | Global financial stability analysis |
| `IMF_WEO_2025_Oct.pdf` | IMF | High | World economic outlook |
| `Fed_Monetary_Policy_Report_*.pdf` (×2) | Federal Reserve | **Highest** | US monetary policy, rate outlook |
| `iShares_ETF_Investing_Guide.pdf` | BlackRock/iShares | High | ETF investing mechanics |
| `arXiv_*.pdf` (×7) | arXiv q-fin | High | Quantitative portfolio theory |

### Requires Manual Browser Download

| Document | URL | Priority |
|----------|-----|----------|
| JP Morgan Guide to the Markets | https://am.jpmorgan.com/us/en/asset-management/adv/insights/market-insights/guide-to-the-markets/ | **#1 Highest** |
| Vanguard Principles for Investing Success | https://institutional.vanguard.com/insights | **#2** |
| Vanguard Advisor's Alpha | https://advisors.vanguard.com/advisors-alpha | High |
| Vanguard Economic and Market Outlook 2026 | https://institutional.vanguard.com/insights | High |
| Schwab Asset Allocation Guide | https://schwab.com/learn/category/investing | Medium |
| SSRN: Fama-French Three-Factor Model | https://ssrn.com/abstract=3748095 | Medium |
| SSRN: Shiller CAPE/Valuation | https://ssrn.com/abstract=2899101 | Medium |
| SSRN: Thaler Behavioral Finance | https://ssrn.com/abstract=3177539 | Medium |
| NBER: Equity Premium Puzzle | https://www.nber.org/papers/w2343 | Medium |
| NBER: Long-Run Stock Returns | https://www.nber.org/papers/w5056 | Medium |

---

## Setup Instructions

### Step 1: Download PDFs

```bash
cd rag_knowledge_base
pip install requests tqdm
python download_pdfs.py
```

This downloads ~20 PDFs automatically. Failed items (gated documents) are printed at the end with manual URLs.

### Step 2: Manual Downloads

Visit each URL in the table above and save PDFs to the appropriate subdirectory. The JP Morgan Guide to the Markets is the single most valuable document — prioritize it.

### Step 3: Ingest into Vector Store

See the RAG ingestion recommendations below.

---

## RAG Ingestion Recommendations

### Chunking Strategy

| Document Type | Chunk Size | Overlap | Rationale |
|--------------|------------|---------|-----------|
| Regulatory guides (SEC/FINRA) | 512 tokens | 64 tokens | Dense prose; moderate overlap for context |
| Central bank reports (BIS/IMF/Fed) | 768 tokens | 128 tokens | Long analytical sections; larger chunks preserve context |
| Academic papers (arXiv) | 512 tokens | 128 tokens | Technical content; high overlap for formula/definition spans |
| Institutional research (JPM/Vanguard) | 400 tokens | 64 tokens | Often structured with tables/charts; smaller chunks |

Use **sentence-boundary-aware splitting** (e.g., `RecursiveCharacterTextSplitter` in LangChain, or `SentenceSplitter` in LlamaIndex) rather than hard character counts.

### Metadata Schema

Every chunk should carry these fields in the vector store:

```json
{
  "doc_id": "SEC_ETF_Guide",
  "source_url": "https://www.investor.gov/introduction-investing/investing-basics/investment-products/exchange-traded-funds-etfs",
  "authority": "regulatory",
  "issuer": "SEC",
  "topic_tags": ["ETF", "passive investing", "tax efficiency", "index funds"],
  "date_published": "2024",
  "date_ingested": "2026-05-30",
  "file_format": "txt",
  "chunk_index": 3,
  "total_chunks": 12
}
```

**Authority levels** (use for retrieval re-ranking):
- `regulatory` — SEC, FINRA, CFPB, FDIC (highest trust; use for compliance/definitional queries)
- `central_bank` — Federal Reserve, BIS, IMF (use for macro/rate/economic context)
- `institutional` — Vanguard, JP Morgan, BlackRock (use for strategy/allocation guidance)
- `academic` — arXiv, SSRN, NBER (use for quantitative methodology queries)

### Retrieval Design

**Query routing suggestions for your multi-agent system:**

- Sharpe ratio / risk-adjusted return questions → `academic` + `institutional`
- Asset allocation for a given risk tolerance → `regulatory` (SEC) + `institutional`
- Current macro/rate environment → `central_bank` (BIS/IMF/Fed)
- ETF vs mutual fund selection → `regulatory` (SEC ETF/MF guides)
- Bond allocation → `regulatory` (SEC Bonds) + `central_bank` (BIS)
- Behavioral coaching / investor discipline → `institutional` (Vanguard)

**Hybrid search** (BM25 + dense vector) is recommended. The SEC/FINRA documents use precise regulatory terminology that keyword search handles better than pure embedding similarity.

### Suggested Embedding Models

- `text-embedding-3-large` (OpenAI) — strong on financial terminology
- `voyage-finance-2` (Voyage AI) — domain-specialized for finance
- `BAAI/bge-large-en-v1.5` — strong open-source alternative

---

## Document Authority Notes

The SEC, FINRA, CFPB, and FDIC documents are **primary regulatory sources** — treat them as ground truth for definitions (what is an ETF, what is a bond, etc.) and risk disclosures.

The BIS Quarterly Reviews, IMF GFSR, and Federal Reserve Monetary Policy Reports are **current macro context** — they reflect conditions as of late 2025/early 2026 and should be tagged with their publication date so the advisory agent can appropriately caveat time-sensitive recommendations.

The JP Morgan Guide to the Markets is a **quarterly data-rich reference** containing valuation charts, asset class returns, and allocation frameworks. It is the most cited institutional document for advisors and should be re-downloaded each quarter.

Vanguard's **Advisor's Alpha** and **Principles for Investing Success** are consensus documents in the RIA community — particularly useful for behavioral coaching context.

The arXiv papers provide **mathematical grounding** for portfolio optimization concepts (Sharpe ratio interpretation, CVaR, risk parity, factor exposures) that your metrics-computation agent will produce.

---

## Maintenance

- **Quarterly**: Re-download JP Morgan Guide to the Markets, BIS Quarterly Review
- **Semi-annual**: Re-run `download_pdfs.py` to pick up new IMF/Fed reports
- **Annual**: Refresh Vanguard Economic Outlook, BIS Annual Report

---

*Knowledge base assembled 2026-05-30. Economic indicators in `FedStLouis_Economic_Overview.txt` reflect May 2026 data.*
