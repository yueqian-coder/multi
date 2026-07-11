# ClaimScope Design System

Generated from UI/UX Pro Max `2.10.2` recommendations and adapted to ClaimScope's Streamlit stack, evidence-first semantics, and operational research workflow.

## Product Frame

- Product type: scientific research operations workbench
- Audience: researchers auditing a fuzzy direction before committing to an idea
- Style: quiet, dense, trustworthy, traceable
- Design dials: variance `4/10`, motion `2/10`, density `7/10`
- Primary task: compare claims, inspect uncertainty, trace evidence, export artifacts

## Design Principles

1. Evidence before decoration.
2. Status is communicated by text and color, never color alone.
3. Dense information remains scannable through alignment and stable dimensions.
4. Synthetic fixtures, uncertainty, and degraded execution stay visually prominent.
5. Motion never blocks work and respects `prefers-reduced-motion`.

## Tokens

| Role | Value | Usage |
|---|---|---|
| Canvas | `#F3F6F7` | page background |
| Surface | `#FFFFFF` | work panels |
| Ink | `#17212B` | primary text |
| Muted | `#5C6A73` | metadata and helper text |
| Border | `#D6DFE2` | dividers and panel boundaries |
| Navy | `#244B5A` | headings and brand hierarchy |
| Teal | `#0B6F70` | primary action and selected state |
| Blue | `#3D5A80` | agents and process state |
| Green | `#277A57` | complete state |
| Amber | `#996300` | limitations and warnings |
| Red | `#B42318` | failure and high risk |

Typography uses the local system stack: `Inter`, `Segoe UI`, and platform sans-serif fallbacks. This avoids a render-blocking font request and keeps the offline demo reproducible. Letter spacing remains `0` across the interface.

## Geometry

- Panel radius: `6px`
- Control radius: `4px`
- Panel padding: `16px`
- Dense gap: `8px`
- Standard gap: `12-16px`
- Minimum interactive height: `44px`
- Shadow: `0 1px 2px rgba(23,33,43,.04)`
- Desktop content width: up to `1500px`

## Component Rules

### Header

Brand, mode control, provider state, and settings share one compact row. At mobile widths they stack without hiding provider consent or the active mode.

### Core Claim

The selected claim is the first result artifact. Structural score, execution mode, falsification test, and open slots have stable metric cells. The long falsification text uses a smaller body size instead of resizing the grid.

### Workflow Rail

Seven numbered stages appear in one horizontal rail. Each stage includes a textual status. Narrow screens scroll the rail horizontally rather than compressing labels into unreadable columns.

### Assumptions And Opportunities

Repeated artifacts use shallow bordered rows, not nested cards. Status, risk, evidence count, opportunity type, and score use compact text badges. Next steps are separated by a divider.

### Warnings

Warnings use amber border, pale amber surface, and explicit text. Failure uses red text plus a status word. Fixture content must always include “demo data, not research evidence.”

## Responsive Rules

- `1440x900`: two-column workbench, four metrics
- `768x1024`: two-column workbench, metrics switch to `2x2`
- `390x844`: stacked header and workbench, metrics remain `2x2`
- Long words and claims use `overflow-wrap:anywhere`
- No fixed viewport-width font scaling
- No content overlap or hidden primary actions

## Accessibility

- Text contrast target: WCAG AA
- Visible `3px` focus ring on buttons, tabs, text areas, and inputs
- Minimum `44px` button height
- Status never relies on red/green alone
- Reduced-motion media query removes nonessential transitions
- Tabs retain text labels and keyboard semantics from Streamlit

## Anti-Patterns

- No gradients, glassmorphism, decorative blobs, or oversized hero copy
- No marketing layout or floating section cards
- No emoji used as controls
- No hidden uncertainty or unmarked synthetic data
- No layout-shifting hover transforms
- No external font dependency

## QA Checklist

- [x] 390px mobile screenshot
- [x] 768px tablet screenshot
- [x] 1440px desktop screenshot
- [x] Full Discovery evidence screenshot
- [x] Visible focus styles
- [x] Reduced-motion rule
- [x] Synthetic fixture warning
- [x] No gradients
- [x] No overlap in tested viewports
