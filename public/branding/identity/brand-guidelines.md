# MedVision-AI Brand Guidelines

**Version 1.0 · Last updated 2026-10-31**

This document defines the visual identity of **MedVision-AI**, an AI-powered Smart Health Analysis and Recommendation System. Use these guidelines to keep our brand consistent across products, marketing, documentation, and partner materials.

---

## 1. Brand Story

MedVision-AI turns raw medical and wellness data into clear, actionable insight. Our visual language blends the calm trust of clinical blue with the vitality of health green and the confident spark of an intelligence layer. We exist to help clinicians, insurers, and individuals see health earlier, clearer, and act faster.

The mark combines three ideas:
- **The pulse line** — continuous health monitoring.
- **The eye dot** — vision, observation, AI attention.
- **The rounded shield** — safety, trust, and care.

---

## 2. Logo

The primary lockup is the **logomark + wordmark** in horizontal form. The logomark may be used alone when the brand is already established in context.

### 2.1 Clear space
Maintain clear space equal to the height of the "M" in "MedVision" on every side of the logo. Do not let any element encroach on this space.

### 2.2 Minimum size
| Format | Minimum |
|---|---|
| Print | 32 mm wide |
| Digital | 160 px wide |
| Favicon / icon | 32 px (logomark only) |

### 2.3 Variants
| Variant | File | Use |
|---|---|---|
| Full color | `logo.svg` | Default, light backgrounds |
| Primary | `logo-primary.svg` | Clinical / professional contexts |
| Secondary | `logo-secondary.svg` | Wellness / lifestyle contexts |
| Icon | `logo-icon.svg` | App tiles, social avatars |
| White | `logo-white.svg` | Dark backgrounds, photography overlays |
| Dark | `logo-dark.svg` | Dark-mode interfaces |
| Favicon | `favicon.svg` | Browser tab, PWA icon |

### 2.4 Misuse — do not
- Stretch, skew, or rotate the logo.
- Change the lockup colors outside the palette.
- Place the full-color logo on busy photography without a scrim.
- Add drop shadows, bevels, or outlines.
- Recreate the logomark with different iconography.

---

## 3. Color Palette

| Role | Token | Hex | RGB | Use |
|---|---|---|---|---|
| Primary | `brand.primary` | `#0EA5E9` | 14, 165, 233 | Primary actions, links, brand moments |
| Secondary | `brand.secondary` | `#10B981` | 16, 185, 129 | Positive results, success, growth |
| Accent | `brand.accent` | `#8B5CF6` | 139, 92, 246 | AI features, intelligence cues |
| Warning | `brand.warning` | `#F59E0B` | 245, 158, 11 | Caution, attention required |
| Danger | `brand.danger` | `#EF4444` | 239, 68, 68 | Critical risk, errors |
| Ink | `ink.900` | `#0F172A` | 15, 23, 42 | Headlines, primary text |
| Slate | `ink.500` | `#64748B` | 100, 116, 139 | Body, secondary text |
| Mist | `surface.200` | `#E2E8F0` | 226, 232, 240 | Borders, dividers |
| Cloud | `surface.50` | `#F8FAFC` | 248, 250, 252 | Page background |
| Night | `surface.900` | `#0F172A` | 15, 23, 42 | Dark mode background |

### 3.1 Color ratios
- Use primary or secondary as the dominant color in any composition. Accent and warning should never together cover more than 25% of any view.
- All text-on-color combinations must meet WCAG AA contrast (4.5:1 for body text).

### 3.2 Data visualization palette
When charts use more than two series, follow this ordered sequence so colorblind users can still read them:

`#0EA5E9 · #10B981 · #8B5CF6 · #F59E0B · #EF4444 · #64748B`

---

## 4. Typography

### 4.1 Primary family — Inter
Inter is our default sans-serif for product UI, marketing, and documentation. It carries the brand voice: precise, calm, modern.

| Style | Use |
|---|---|
| Inter Regular 400 | Body copy |
| Inter Medium 500 | Subheads, labels, captions |
| Inter Semibold 600 | Buttons, tags |
| Inter Bold 700 | Headlines, KPIs |

### 4.2 Display family — Sora
Sora is used for hero headlines, presentation covers, and any moment that needs a little extra character.

### 4.3 Mono family — JetBrains Mono
Used only for code, API responses, log output, and any place where character alignment matters.

### 4.4 Type scale
| Token | Size | Line height | Use |
|---|---|---|---|
| `display` | 56 / 64 | Hero headlines |
| `h1` | 40 / 48 | Page titles |
| `h2` | 32 / 40 | Section titles |
| `h3` | 24 / 32 | Subsection titles |
| `body-lg` | 18 / 28 | Lead paragraphs |
| `body` | 16 / 24 | Default body |
| `caption` | 13 / 20 | Captions, meta |
| `overline` | 12 / 16 (uppercase, tracking 0.12em) | Section labels |

---

## 5. Iconography

Our icons are 48×48 SVG line icons with 2.4 px stroke, rounded caps and joins, drawn on a 4 px grid. Use the matching semantic color from the palette:

- Dashboard → `#0EA5E9`
- Health → `#EF4444`
- AI → `#8B5CF6`
- Analytics → `#10B981`
- Recommendation → `#F59E0B`

Never fill icons (except for the dot in `recommendation` and the logomark). Never apply gradient strokes.

---

## 6. Imagery

- Prefer soft, naturally lit photography of clinicians and patients using devices.
- For abstract concepts, use our flat illustration style: 2-color, soft gradients, rounded geometry.
- Avoid stock imagery of stethoscopes on laptops, floating hearts, or any clichéd medical stock photo.

---

## 7. Voice and tone

- **Clear before clever.** A user reading "your blood pressure is trending up" should never have to decode a pun.
- **Specific, not generic.** "A 12% drop in weekly resting heart rate" beats "great progress."
- **Calm, never alarmist.** Even when surfacing a critical risk, lead with the observation, then the action.
- **Inclusive and human.** Speak to people, not "users," "patients," or "members" when possible.

---

## 8. Component cues

- Cards use 16 px radius, 1 px `#E2E8F0` border, no shadow at rest. On hover, add `0 1px 2px rgba(15,23,42,0.04)` and lift 1 px.
- Buttons: 8 px radius, 40 px height, 16 px horizontal padding. Primary buttons fill with `#0EA5E9`; secondary buttons use `#0EA5E9` text on `#F1F5F9` fill.
- Badges use the corresponding palette color at 12% opacity as background, full color as text.

---

## 9. Accessibility

- All interactive elements meet WCAG 2.2 AA.
- Color is never the sole carrier of meaning. Pair color with an icon, label, or pattern.
- Touch targets are at least 44×44 px.

---

## 10. Contact

Brand questions, asset requests, and partnership co-branding: **brand@medvision-ai.example**. Approval turnaround is typically three business days.
