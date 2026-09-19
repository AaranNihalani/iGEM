"""Headless-browser test of the built explorer page (Playwright/Chromium). Run: python web/test_page.py"""
import sys, json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = (ROOT / "report" / "substrate_predictor_explorer.html").as_uri()
OUT = Path("/tmp/spshots"); OUT.mkdir(exist_ok=True)
errors = []

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={"width": 1440, "height": 1000})
    pg.on("console", lambda m: errors.append(("console." + m.type, m.text)) if m.type in ("error",) else None)
    pg.on("pageerror", lambda e: errors.append(("pageerror", str(e))))
    pg.goto(URL); pg.wait_for_timeout(2500)
    print("title:", pg.title())
    print("chips:", pg.locator("#chips .chip").count(), "| panel chips:", pg.locator("#pcs .pc").count())
    print("verdict big:", pg.inner_text("#vbig").replace("\n", " "), "|", pg.inner_text("#vtxt")[:140].replace("\n", " "))
    print("canvases in v1/v2:", pg.locator("#v1 canvas").count(), pg.locator("#v2 canvas").count())
    pg.screenshot(path=str(OUT / "full.png"), full_page=True)
    # engine cross-check against Python for the A11 core in the wound panel
    r = pg.evaluate("""() => { const S = window.__SP; const p = S.E.parseInput('SGKIPRT|LTA'); const sc = S.E.scorePanel(S.MODELS, ['MMP2','MMP8','MMP9','NE','LASB'], p.window); return {w: p.window, u: Object.fromEntries(Object.entries(sc).map(([k,v]) => [k, v.u]))}; }""")
    print("A11 window", r["w"], {k: round(v, 3) for k, v in r["u"].items()})
    # interactions
    pg.fill("#pep", "AAPV|SGIG"); pg.wait_for_timeout(400)
    pg.locator("#pcs .pc", has_text="NE").locator("button").click(); pg.wait_for_timeout(600)
    print("NE target verdict:", pg.inner_text("#vtxt")[:160].replace("\n", " "))
    pg.screenshot(path=str(OUT / "ne.png"), full_page=True)
    pg.fill("#pep", "IPRTLTAX"); pg.wait_for_timeout(300)
    print("gapped input msg:", pg.inner_text("#pmsg") or "(none)", "| window cells unknown:", pg.locator("#win .cell.unk").count())
    pg.fill("#pep", "SGK"); pg.wait_for_timeout(300)
    print("bad input msg:", pg.inner_text("#pmsg"))
    pg.fill("#pep", "GPRQITA-"); pg.wait_for_timeout(300)
    # scan tab + click a cell
    pg.click("#scantabs button[data-m=dMargin]"); pg.wait_for_timeout(200)
    n = pg.locator("#scan .sc").count(); print("scan cells:", n)
    pg.locator("#scan .sc").nth(5).click(); pg.wait_for_timeout(300); print("after cell click input:", pg.input_value("#pep"))
    # panel preset + structure select
    pg.select_option("#preset", "mmp9_ne"); pg.wait_for_timeout(300); print("mmp9_ne members:", pg.locator("#pcs .pc:not(.off)").count())
    pg.select_option("#structsel", "LASB"); pg.wait_for_timeout(800)
    pg.click("#b-surf"); pg.wait_for_timeout(1500)
    pg.screenshot(path=str(OUT / "final.png"), full_page=True)
    # mobile
    m = b.new_page(viewport={"width": 400, "height": 900}); m.on("pageerror", lambda e: errors.append(("mobile.pageerror", str(e)))); m.goto(URL); m.wait_for_timeout(2000)
    print("mobile horizontal overflow:", m.evaluate("document.documentElement.scrollWidth > document.documentElement.clientWidth"))
    m.screenshot(path=str(OUT / "mobile.png"), full_page=True)
    b.close()
print("ERRORS:", errors if errors else "none")
sys.exit(1 if errors else 0)
