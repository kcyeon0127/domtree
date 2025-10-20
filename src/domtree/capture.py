"""Utilities for capturing rendered webpages and saving screenshots/HTML."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

from playwright.sync_api import Playwright, TimeoutError as PlaywrightTimeoutError, sync_playwright

logger = logging.getLogger(__name__)


DEFAULT_COOKIE_TEXTS = (
    "accept",
    "agree",
    "consent",
    "확인",
    "동의",
    "수락",
)

DEFAULT_COOKIE_SELECTORS = (
    "button[aria-label*='accept']",
    "button[aria-label*='consent']",
    "button[aria-label*='동의']",
    "#onetrust-accept-btn-handler",
    ".[id*='cookie'] button",
)


@dataclasses.dataclass
class CaptureOptions:
    """Configuration parameters for Playwright screenshot capture."""

    output_dir: Path = Path("data/screenshots")
    wait_until: str = "networkidle"
    viewport_width: int = 1440
    viewport_height: int = 900
    timeout_ms: int = 30_000
    close_cookie_popups: bool = True
    cookie_texts: Sequence[str] = dataclasses.field(default_factory=lambda: DEFAULT_COOKIE_TEXTS)
    cookie_selectors: Sequence[str] = dataclasses.field(default_factory=lambda: DEFAULT_COOKIE_SELECTORS)
    auto_scroll: bool = True
    scroll_delay: float = 0.4
    max_scroll_steps: int = 40
    scroll_increment: int = 1_000
    wait_after_load: float = 1.0
    inject_custom_script: Optional[str] = None
    annotate_bounding_boxes: bool = True

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


def _annotate_dom(page) -> dict:
    """Tag DOM nodes with bounding boxes and record viewport size."""

    return page.evaluate(
        """
        () => {
            const viewportWidth = window.innerWidth;
            const viewportHeight = window.innerHeight;
            document
                .querySelectorAll('[data-domtree-bbox]')
                .forEach(el => el.removeAttribute('data-domtree-bbox'));
            const elements = document.querySelectorAll('*');
            elements.forEach(el => {
                const rect = el.getBoundingClientRect();
                el.setAttribute(
                    'data-domtree-bbox',
                    `${rect.top},${rect.left},${rect.bottom},${rect.right}`
                );
            });
            if (document.body) {
                document.body.setAttribute(
                    'data-domtree-viewport',
                    `${viewportWidth},${viewportHeight}`
                );
            }
            return {
                viewportWidth,
                viewportHeight,
            };
        }
        """
    )


def _close_cookie_popups(page, *, texts: Iterable[str], selectors: Iterable[str]) -> None:
    """Attempt to close cookie consent dialogs using heuristics."""

    # Try selectors first for fast match
    for selector in selectors:
        with contextlib.suppress(PlaywrightTimeoutError):
            elements = page.locator(selector)
            if elements.count() > 0:
                logger.debug("Attempting to click cookie element via selector %s", selector)
                elements.first.click(timeout=2_000)
                page.wait_for_timeout(300)
                return

    # Fallback: find buttons by text
    lowered = [text.lower() for text in texts]
    buttons = page.locator("button, text=*")
    count = buttons.count()
    for idx in range(count):
        button = buttons.nth(idx)
        with contextlib.suppress(PlaywrightTimeoutError):
            label = button.inner_text(timeout=1_000).strip().lower()
        if not label:
            continue
        if any(text in label for text in lowered):
            logger.debug("Clicking cookie button with label '%s'", label)
            with contextlib.suppress(PlaywrightTimeoutError):
                button.click(timeout=2_000)
                page.wait_for_timeout(300)
                return


def _auto_scroll(page, *, options: CaptureOptions) -> None:
    """Scroll the page to force lazy content to render."""

    if not options.auto_scroll:
        return

    page_height = page.evaluate("() => document.body.scrollHeight")
    logger.debug("Initial scroll height reported as %s", page_height)
    scroll_pos = 0
    steps = 0
    while steps < options.max_scroll_steps and scroll_pos < page_height:
        scroll_pos = min(scroll_pos + options.scroll_increment, page_height)
        page.evaluate("pos => window.scrollTo({ top: pos, behavior: 'instant' })", scroll_pos)
        steps += 1
        logger.debug("Scrolled to position %s (step %s)", scroll_pos, steps)
        page.wait_for_timeout(int(options.scroll_delay * 1_000))
        page_height = page.evaluate("() => document.body.scrollHeight")

    page.evaluate("() => window.scrollTo({ top: 0, behavior: 'instant' })")


def capture_page(url: str, *, options: Optional[CaptureOptions] = None, name: Optional[str] = None) -> dict:
    """Capture a webpage screenshot and rendered HTML using Playwright.

    Returns a dictionary with keys `screenshot_path` and `html_path`.
    """

    options = options or CaptureOptions()
    options.ensure_dirs()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_name = name or url.replace("https://", "").replace("http://", "").replace("/", "_")
    screenshot_path = options.output_dir / f"{safe_name}_{timestamp}.png"
    html_path = options.output_dir / f"{safe_name}_{timestamp}.html"

    logger.info("Capturing %s", url)
    viewport_meta = {"viewportWidth": None, "viewportHeight": None}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={
                "width": options.viewport_width,
                "height": options.viewport_height,
            },
        )
        page = context.new_page()
        page.goto(url, wait_until=options.wait_until, timeout=options.timeout_ms)

        if options.wait_after_load:
            logger.debug("Waiting %ss after initial load", options.wait_after_load)
            page.wait_for_timeout(int(options.wait_after_load * 1_000))

        if options.close_cookie_popups:
            with contextlib.suppress(Exception):
                _close_cookie_popups(page, texts=options.cookie_texts, selectors=options.cookie_selectors)

        if options.inject_custom_script:
            logger.debug("Injecting custom JavaScript")
            page.evaluate(options.inject_custom_script)

        _auto_scroll(page, options=options)

        if options.annotate_bounding_boxes:
            with contextlib.suppress(Exception):
                viewport_meta = _annotate_dom(page)

        page.screenshot(path=str(screenshot_path), full_page=True)
        html_content = page.content()
        html_path.write_text(html_content, encoding="utf-8")

        browser.close()

    logger.info("Saved screenshot to %s", screenshot_path)
    return {
        "screenshot_path": screenshot_path,
        "html_path": html_path,
        "viewport": viewport_meta,
    }


def fetch_rendered_html(url: str, *, options: Optional[CaptureOptions] = None) -> str:
    """Return the rendered HTML for a given URL without saving screenshots."""

    options = options or CaptureOptions()
    options.wait_after_load = max(options.wait_after_load, 0.5)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={
                "width": options.viewport_width,
                "height": options.viewport_height,
            }
        )
        page = context.new_page()
        page.goto(url, wait_until=options.wait_until, timeout=options.timeout_ms)
        if options.wait_after_load:
            page.wait_for_timeout(int(options.wait_after_load * 1_000))
        if options.close_cookie_popups:
            with contextlib.suppress(Exception):
                _close_cookie_popups(page, texts=options.cookie_texts, selectors=options.cookie_selectors)
        html = page.content()
        browser.close()
    return html
