# LTX-2.5 해상도 선택기 — 16:9 · multiple=64 · 공식 재계산

공식: `w = ceil(sqrt(MP×10⁶×AR)/64)×64`, `h = ceil(sqrt(MP×10⁶/AR)/64)×64` (AR = 16/9)

```markdown
| megapixels | Output (multiple=64, 공식) | stage1 (÷2) |
|---|---|---|
| 0.2 | 640 × 384 | 320 × 192 |
| 0.3 | 768 × 448 | 384 × 224 |
| 0.4 | 896 × 512 | 448 × 256 |
| 0.5 | 960 × 576 | 480 × 288 |
| 0.6 | 1088 × 640 | 544 × 320 |
| 0.7 | 1152 × 640 | 576 × 320 |
| 0.8 | 1216 × 704 | 608 × 352 |
| 0.9 | 1280 × 768 | 640 × 384 |
| 0.98 | 1344 × 768 | 672 × 384 |
| 1.0 | 1344 × 768 | 672 × 384 |
| 1.2 | 1472 × 832 | 736 × 416 |
| 1.5 | 1664 × 960 | 832 × 480 |
| 1.8 | 1792 × 1024 | 896 × 512 |
| 2.0 | 1920 × 1088 | 960 × 544 |
```

## 구현 상태 (2026-08-23)

- **우리 노드**: `comfyui-ltx25-resolution` 팩의 `LTX25Controls` ("LTX25 Smart Controls (full)") 및
  `LTX25Resolution` ("LTX25 Resolution Selector")
  - aspect_ratio (8종 + Custom (manual)) / **stage1_megapixels + stage2_megapixels + final_megapixels** (각 0.2~2.0 셀렉트) / multiple(16/32/64/128/256 셀렉트)
  - **3단계 해상도 (미리보기 → 중간 → 최종)**: `stage1_megapixels`(기본 0.2) → stage1_width/height,
    `stage2_megapixels`(기본 0.5) → stage2_width/height, `final_megapixels`(기본 0.98) → final_width/height.
    워크플로우에서 1차는 stage1 치수(빠른 미리보기), 후보 축소 후 stage2, 최종 선택 후엔 final 치수로 생성.
  - Custom (manual)이면 final = custom_width/height, stage2 = final//2, stage1 = stage2//2 (snap 유지, legacy)
  - **final_width/final_height는 입력이 아닌 계산 출력** — aspect_ratio ≠ Custom이면
    `snap(ceil(sqrt(MP×10⁶×AR)/multiple)×multiple)`로 자동계산, Custom이면 `custom_width`/`custom_height` 위젯 입력 사용
  - 스냅은 선택된 `multiple` 기준 (multiple=32 → 32 그리드 = MiniMax H3, multiple=64 → LTX Conv VAE 64 그리드)
  - 출력: final_width, final_height, stage1_width, stage1_height, frames, … , resolved_summary, stage2_width, stage2_height
  - 서드파티 팩과 독립 → 업데이트에 안전
- **서드파티 팩 원복**: `ltx25_smart_controls`는 원본 복구 (aspect_ratio 위젯 제거)
- **워크플로우 배선** (수동 필요): MiniMaxH3ImageToVideo / EmptyMiniMaxH3LatentAV 의
  width ← `final_width`, height ← `final_height`, length ← `frames` 출력에 연결.
  기존에 408의 final_width/height 입력으로 들어가던 링크는 입력 제거로 끊어짐.
- **2단계 파이프라인 예시**: 1차 미리보기 Sampler ← stage1_width/height (+ 낮은 스텝 시그마 3step),
  선택된 latent → 2차 고해상도 Sampler ← stage2_width/height (+ 4step+). stage1 치수는 메가픽셀 절감으로
  빨라지고, 최종 출력만 stage2 해상도로 생성.
