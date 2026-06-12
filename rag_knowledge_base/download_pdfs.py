#!/usr/bin/env python3
"""
Financial RAG Knowledge Base — PDF Downloader
=============================================
Run this script to download all high-priority PDF documents for your
financial advisory RAG system. Some documents (JP Morgan, Vanguard, iShares)
require visiting the source website directly due to access controls.

Usage:
    cd rag_knowledge_base/
    python download_pdfs.py

PDFs are saved to ./{category}/pdfs/ (e.g., academic/pdfs/, central_bank/pdfs/).
After downloading, run: cd ../backend && uv run python scripts/seed_kb.py --force
"""

import httpx
from pathlib import Path

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

DOCUMENTS = {

    # ─── PRIORITY 1: Regulatory / Government ─────────────────────────────────
    # SEC.gov, FINRA, CFPB, FDIC PDFs are blocked by Akamai/Cloudflare (403) or
    # have moved (404). Their content is covered by the authoritative .txt files
    # in regulatory/ and central_bank/. Download manually if needed (see below).
    "regulatory": [
        {
            "name": "SEC_Mutual_Funds_ETFs_Guide",
            "url": "https://www.sec.gov/investor/pubs/sec-guide-to-mutual-funds.pdf",
            "desc": "SEC Guide to Mutual Funds and ETFs"
        },
        # BIS Basel III regulatory capital framework — definitive risk standard
        {
            "name": "BIS_Basel_III_Final_Rules",
            "url": "https://www.bis.org/bcbs/publ/d424.pdf",
            "desc": "Basel III: Finalising Post-Crisis Reforms (BIS BCBS d424)"
        },
    ],

    # ─── PRIORITY 2: Central Banks (confirmed working) ────────────────────────
    "central_bank": [
        {
            "name": "BIS_Quarterly_Review_2025_Sep",
            "url": "https://www.bis.org/publ/qtrpdf/r_qt2509.pdf",
            "desc": "BIS Quarterly Review September 2025"
        },
        {
            "name": "BIS_Quarterly_Review_2025_Dec",
            "url": "https://www.bis.org/publ/qtrpdf/r_qt2512.pdf",
            "desc": "BIS Quarterly Review December 2025"
        },
        {
            "name": "BIS_Quarterly_Review_2026_Mar",
            "url": "https://www.bis.org/publ/qtrpdf/r_qt2503.pdf",
            "desc": "BIS Quarterly Review March 2026"
        },
        {
            "name": "BIS_Annual_Report_2025",
            "url": "https://www.bis.org/publ/arpdf/ar2025e.pdf",
            "desc": "BIS Annual Economic Report 2025"
        },
        # Federal Reserve Monetary Policy Reports (Feb = /publications/, Jun = /monetarypolicy/)
        {
            "name": "Fed_Monetary_Policy_Report_2025_Feb",
            "url": "https://www.federalreserve.gov/publications/files/20250207_mprfullreport.pdf",
            "desc": "Federal Reserve Monetary Policy Report February 2025"
        },
        {
            "name": "Fed_Monetary_Policy_Report_2025_Jun",
            "url": "https://www.federalreserve.gov/monetarypolicy/files/20250620_mprfullreport.pdf",
            "desc": "Federal Reserve Monetary Policy Report June 2025"
        },
        # IMF flagship reports — blocked by Akamai (403); download manually from imf.org
        # IMF_GFSR_2026_Apr → https://www.imf.org/en/Publications/GFSR
        # IMF_WEO_2025_Oct  → https://www.imf.org/en/Publications/WEO
        # Save to: central_bank/pdfs/IMF_GFSR_2026_Apr.pdf and IMF_WEO_2025_Oct.pdf
    ],

    # ─── PRIORITY 3: Institutional Research ──────────────────────────────────
    "institutional_research": [
        # JP Morgan Guide to the Markets — confirmed downloadable
        {
            "name": "JPM_Guide_to_Markets_Q2_2026",
            "url": "https://am.jpmorgan.com/content/dam/jpm-am-aem/global/en/insights/market-insights/guide-to-the-markets/mi-guide-to-the-markets-us.pdf",
            "desc": "JP Morgan Guide to the Markets Q2 2026"
        },
        # Vanguard / iShares require browser authentication — see MANUAL_DOWNLOADS below
    ],

    # ─── PRIORITY 4: Academic (arXiv — all verified q-fin content) ──────────
    # All IDs verified against arXiv abstract pages 2026-06-06.
    # Previous IDs pointed to astrophysics/physics/math papers — all replaced.
    "academic": [
        {
            "name": "arXiv_Portfolio_Optimization_Survey",
            "url": "https://arxiv.org/pdf/2201.06635",
            "desc": "Optimal Trend Following Portfolios — autocorrelation model with covariance of trends and risk premia (Valeyre 2022, q-fin.PM)"
        },
        {
            "name": "arXiv_Risk_Parity_Portfolios",
            "url": "https://arxiv.org/pdf/2203.00148",
            "desc": "Improved Iterative Methods for Risk Parity Portfolio (Choi & Chen 2022, q-fin.PM)"
        },
        {
            "name": "arXiv_Drawdown_Risk_Measures",
            "url": "https://arxiv.org/pdf/2401.02601",
            "desc": "Constrained Max Drawdown: Fast and Robust Portfolio Optimization (Dorador 2024, q-fin.PM)"
        },
        {
            "name": "arXiv_Bayesian_Portfolio_Analysis",
            "url": "https://arxiv.org/pdf/1803.03573",
            "desc": "Bayesian Mean-Variance Analysis: Optimal Portfolio Under Parameter Uncertainty (Bauder et al. 2018, q-fin.PM)"
        },
        {
            "name": "arXiv_Sharpe_Ratio_Estimation",
            "url": "https://arxiv.org/pdf/1906.00573",
            "desc": "Conditional Inference on the Asset with Maximum Sharpe Ratio (Pav 2019, q-fin.PM)"
        },
        {
            "name": "arXiv_CVaR_Portfolio_Optimization",
            "url": "https://arxiv.org/pdf/2103.16451",
            "desc": "Robustifying Conditional Portfolio Decisions via Optimal Transport / CVaR (Nguyen et al. 2021, q-fin.PM)"
        },
        {
            "name": "arXiv_Factor_Investing",
            "url": "https://arxiv.org/pdf/2209.13623",
            "desc": "Publication Bias in Asset Pricing Research: Factor Premiums (Chen & Zimmermann 2022, q-fin.GN)"
        },
        {
            "name": "arXiv_Modern_Portfolio_Theory_Review",
            "url": "https://arxiv.org/pdf/2201.00914",
            "desc": "Continuous-time Markowitz Mean-Variance Model (Guan et al. 2022, q-fin.MF)"
        },
        {
            "name": "arXiv_Risk_Measures_Portfolio",
            "url": "https://arxiv.org/pdf/1609.04065",
            "desc": "Closed-form Worst-case Law Invariant Risk Measures for Robust Portfolio (Li 2016, q-fin.RM)"
        },
        {
            "name": "arXiv_Asset_Allocation_Review",
            "url": "https://arxiv.org/pdf/1910.11840",
            "desc": "Sparsity and Stability for Minimum-Variance Portfolios (q-fin.PM)"
        },
        {
            "name": "arXiv_Bond_Duration_Risk",
            "url": "https://arxiv.org/pdf/1206.6998",
            "desc": "Interest Rate Risk of Bonds — Duration, Modified Duration, Convexity"
        },
        {
            "name": "arXiv_Equity_Risk_Premium",
            "url": "https://arxiv.org/pdf/1903.07737",
            "desc": "Risk and Return Models for Equity Markets and Implied Equity Risk Premium"
        },
        {
            "name": "arXiv_ETF_vs_Index_Funds",
            "url": "https://arxiv.org/pdf/1111.0389",
            "desc": "ETF vs Index Fund Performance Comparison 2002-2010 (q-fin)"
        },
        {
            "name": "arXiv_Active_vs_Passive_Allocation",
            "url": "https://arxiv.org/pdf/1803.05819",
            "desc": "Active and Passive Portfolio Management — Outperformance and Tracking"
        },
        {
            "name": "arXiv_Investor_Behavior_Modeling",
            "url": "https://arxiv.org/pdf/2107.05592",
            "desc": "Investor Behavior Modeling by Analyzing Financial Advisor Notes"
        },
    ],
}

MANUAL_DOWNLOADS = """
=============================================================================
MANUAL DOWNLOAD REQUIRED
These require a browser (auth, JavaScript rendering, or CDN bot protection).
=============================================================================

── INSTITUTIONAL RESEARCH → save to: institutional_research/pdfs/ ──────────

1. Vanguard Principles for Investing Success  [HIGHEST PRIORITY]
   URL: https://institutional.vanguard.com/insights
   → Search "Principles for Investing Success" → Download PDF
   → Save as: Vanguard_Principles_Investing_Success.pdf

2. Vanguard Advisor's Alpha
   URL: https://advisors.vanguard.com/advisors-alpha
   → Save as: Vanguard_Advisors_Alpha.pdf

3. Vanguard Economic and Market Outlook 2026
   URL: https://institutional.vanguard.com/insights
   → Search "Economic and Market Outlook" → Download PDF
   → Save as: Vanguard_Economic_Outlook_2026.pdf

4. iShares ETF Investing Guide (BlackRock)
   URL: https://www.ishares.com/us/education
   → Save as: iShares_ETF_Guide.pdf

5. Schwab Asset Allocation Guide
   URL: https://schwab.com/learn/category/investing
   → Save as: Schwab_Asset_Allocation_Guide.pdf

── CENTRAL BANK → save to: central_bank/pdfs/ ──────────────────────────────

6. IMF Global Financial Stability Report April 2026  [HIGH PRIORITY]
   URL: https://www.imf.org/en/Publications/GFSR
   → Download the April 2026 full report PDF
   → Save as: IMF_GFSR_2026_Apr.pdf

7. IMF World Economic Outlook October 2025
   URL: https://www.imf.org/en/Publications/WEO
   → Download October 2025 full report PDF
   → Save as: IMF_WEO_2025_Oct.pdf

── REGULATORY → save to: regulatory/pdfs/ ──────────────────────────────────
   (Content is largely covered by existing .txt files in regulatory/ and
    central_bank/ — download only if you want the original PDFs)

8. FINRA Smart Bond Investing
   URL: https://www.finra.org/investors/learn-to-invest/types-investments/bonds
   → Save as: FINRA_Smart_Bond_Investing.pdf

9. CFPB Your Money Your Goals Toolkit
   URL: https://www.consumerfinance.gov/consumer-tools/educator-tools/your-money-your-goals/
   → Save as: CFPB_Your_Money_Your_Goals.pdf

── ACADEMIC → save to: academic/pdfs/ ──────────────────────────────────────
   (Free after creating a free account)

10. SSRN: Fama-French Three-Factor Model
    URL: https://ssrn.com/abstract=3748095
    → Save as: SSRN_Fama_French_Three_Factor.pdf

11. SSRN: Shiller CAPE / Valuation
    URL: https://ssrn.com/abstract=2899101
    → Save as: SSRN_Shiller_CAPE.pdf

12. SSRN: Thaler Behavioral Finance
    URL: https://ssrn.com/abstract=3177539
    → Save as: SSRN_Thaler_Behavioral_Finance.pdf

=============================================================================
After adding manual downloads, re-run the seeder:
  cd ../backend && uv run python scripts/seed_kb.py --force
=============================================================================
"""


def download_file(url: str, dest_path: Path, desc: str) -> bool:
    """Download a single file. Returns True on success."""
    try:
        print(f"  Downloading: {desc}")
        with httpx.Client(headers=HEADERS, follow_redirects=True, timeout=60) as client:
            r = client.get(url)
        if r.status_code != 200 or len(r.content) < 2000:
            print(f"  ✗ FAILED: HTTP {r.status_code}, size {len(r.content)} bytes")
            return False
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(r.content)
        size_kb = len(r.content) / 1024
        print(f"  ✓ Saved ({size_kb:.0f} KB): {dest_path.name}")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def main():
    base = Path(__file__).parent
    results = {"success": [], "failed": [], "skipped": []}

    for category, docs in DOCUMENTS.items():
        cat_dir = base / category / "pdfs"
        cat_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"  Category: {category.upper()}")
        print(f"{'='*60}")

        for doc in docs:
            dest = cat_dir / f"{doc['name']}.pdf"
            if dest.exists():
                print(f"  ↷ Already exists (skipping): {doc['name']}")
                results["skipped"].append(doc["name"])
                continue

            ok = download_file(doc["url"], dest, doc["desc"])
            if ok:
                results["success"].append(doc["name"])
            else:
                results["failed"].append(doc["name"])

    print("\n" + "="*60)
    print("  DOWNLOAD SUMMARY")
    print("="*60)
    print(f"  ✓ Success:  {len(results['success'])}")
    print(f"  ↷ Skipped:  {len(results['skipped'])} (already downloaded)")
    print(f"  ✗ Failed:   {len(results['failed'])}")
    if results["failed"]:
        print("\n  Failed documents (may need manual download):")
        for name in results["failed"]:
            print(f"    - {name}")

    print(MANUAL_DOWNLOADS)


if __name__ == "__main__":
    main()
