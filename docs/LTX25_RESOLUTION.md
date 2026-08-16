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

## 구현 상태 (2026-08-16)

- **우리 노드**: `comfyui-ltx25-resolution` 팩의 `LTX25Resolution` ("LTX25 Resolution Selector")
  - aspect_ratio (8종 + Custom) / megapixels / multiple(기본 64)
  - 출력: final_width, final_height, stage1_width, stage1_height, summary
  - 서드파티 팩과 독립 → 업데이트에 안전
- **서드파티 팩 원복**: `ltx25_smart_controls`는 원본 복구 (aspect_ratio 위젯 제거)
- **워크플로우**: 사용자가 코어 `ResolutionSelector`(630)를 408의 final_width/height에 이미 배선
  (코어 노드라 업데이트 안전)
