# Number Audit Report

Generated per Rule 1 and Section 9.4.

## 1. Forbidden Placeholders Check
- **PASS:** Zero forbidden tokens (TODO, TBD, XXX, lorem, ??) detected in report source.

- **PASS:** Zero forbidden buzzwords (delve, crucial, pivotal, robust, leverage, seamless, etc.) detected.

## 2. Prose Number Traceability Audit

| Number | Classification | Status |
| :--- | :--- | :--- |
| `0` | Config / Hardware / Structural Parameter | **VALID** |
| `0.5` | Prose number | **FLAGGED** |
| `0.95` | Config / Hardware / Structural Parameter | **VALID** |
| `1` | Config / Hardware / Structural Parameter | **VALID** |
| `10` | Config / Hardware / Structural Parameter | **VALID** |
| `100` | Prose number | **FLAGGED** |
| `1000` | Config / Hardware / Structural Parameter | **VALID** |
| `119` | Config / Hardware / Structural Parameter | **VALID** |
| `12` | Config / Hardware / Structural Parameter | **VALID** |
| `120` | Config / Hardware / Structural Parameter | **VALID** |
| `2` | Config / Hardware / Structural Parameter | **VALID** |
| `20` | Config / Hardware / Structural Parameter | **VALID** |
| `200` | Config / Hardware / Structural Parameter | **VALID** |
| `2026` | Config / Hardware / Structural Parameter | **VALID** |
| `2027` | Config / Hardware / Structural Parameter | **VALID** |
| `256` | Prose number | **FLAGGED** |
| `29` | Config / Hardware / Structural Parameter | **VALID** |
| `3` | Prose number | **FLAGGED** |
| `3.0` | Config / Hardware / Structural Parameter | **VALID** |
| `4` | Config / Hardware / Structural Parameter | **VALID** |
| `40` | Config / Hardware / Structural Parameter | **VALID** |
| `42` | Config / Hardware / Structural Parameter | **VALID** |
| `5` | Config / Hardware / Structural Parameter | **VALID** |
| `50` | Config / Hardware / Structural Parameter | **VALID** |
| `6` | Config / Hardware / Structural Parameter | **VALID** |
| `63.0` | Prose number | **FLAGGED** |
| `64` | Prose number | **FLAGGED** |
| `8` | Config / Hardware / Structural Parameter | **VALID** |
| `80` | Config / Hardware / Structural Parameter | **VALID** |
| `90` | Config / Hardware / Structural Parameter | **VALID** |

## 3. Dynamic LaTeX Macros Audit

The following metrics are dynamically loaded from `report/generated/numbers.tex`:

- `\newcommand{\totalFramesCount}{120}`
- `\newcommand{\frameWidth}{640}`
- `\newcommand{\frameHeight}{360}`
- `\newcommand{\videoFps}{24}`
- `\newcommand{\maxitStart}{200}`
- `\newcommand{\maxitEnd}{1000}`
- `\newcommand{\zoomCenterX}{-0.7436}`
- `\newcommand{\zoomCenterY}{0.1318}`
- `\newcommand{\sTwoImbalanceRatio}{1.03}`
- `\newcommand{\sThreeBlockImbalanceRatio}{1.48}`
- `\newcommand{\sThreeCyclicImbalanceRatio}{1.01}`
- `\newcommand{\serialMedianTime}{40.41}`
- `\newcommand{\serialFps}{2.97}`
- `\newcommand{\threadsVariantABestP}{12}`
- `\newcommand{\threadsVariantABestTime}{5.42}`
- `\newcommand{\threadsVariantABestSpeedup}{7.46}`
- `\newcommand{\threadsVariantBBestP}{12}`
- `\newcommand{\threadsVariantBBestTime}{4.83}`
- `\newcommand{\threadsVariantBBestSpeedup}{8.37}`
- `\newcommand{\mpiSOneBestP}{12}`
- `\newcommand{\mpiSOneBestTime}{4.94}`
- `\newcommand{\mpiSOneBestSpeedup}{8.18}`
- `\newcommand{\mpiSTwoBestP}{12}`
- `\newcommand{\mpiSTwoBestTime}{5.47}`
- `\newcommand{\mpiSTwoBestSpeedup}{7.38}`
- `\newcommand{\mpiSThreeCyclicBestP}{12}`
- `\newcommand{\mpiSThreeCyclicBestTime}{3.96}`
- `\newcommand{\mpiSThreeCyclicBestSpeedup}{10.20}`
- `\newcommand{\mpiSThreeBlockBestP}{12}`
- `\newcommand{\mpiSThreeBlockBestTime}{5.36}`
- `\newcommand{\mpiSThreeBlockBestSpeedup}{7.54}`
- `\newcommand{\sTwoOptimalTileSize}{80}`
- `\newcommand{\sTwoOptimalTileTime}{13.83}`
- `\newcommand{\pingpongMinLatencyMicrosec}{2.24}`
- `\newcommand{\pingpongMaxBandwidthMbps}{3882.47}`
- `\newcommand{\pingpongMaxBandwidthGbps}{3.88}`
- `\newcommand{\productionRenderTime}{7.70}`
- `\newcommand{\productionIoTime}{10.62}`
- `\newcommand{\productionTotalTime}{18.32}`
