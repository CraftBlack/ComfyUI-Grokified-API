# ComfyUI Grokified API

Node ComfyUI untuk memakai **Grokified** (`https://api.grokified.com/v1`) — API
OpenAI-compatible — sebagai pengganti LLM lokal. Dirancang sebagai **drop-in**
untuk node `LLM Text Processor` dari `ComfyUI-LLM-text-processor`:
output `RESPONSE` / `REASONING` / `PERF` ada di slot yang sama (0/1/2), input
`image` dan `prompt` juga sama, jadi workflow yang sudah ada cuma perlu ganti tipe node.

## Kenapa node ini ada

Node `ComfyUI-OpenAI-API` (`OpenAIChatCompletion`) sebenarnya **bisa** bicara dengan
Grokified karena wire protocol-nya OpenAI-compatible, tapi ada beberapa hal yang bikin
tidak nyaman (hasil probe langsung ke API, lihat `tests/`):

| Temuan probe | Dampak ke `ComfyUI-OpenAI-API` |
|---|---|
| `seed: 0` → `400 Seed must be positive but seed = 0` | **Gagal total secara default** (`use_seed=True`, `seed=0`) |
| `stop` tidak didukung `grok-4.7` | Tidak dipakai node itu, aman |
| `reasoning_content` ada di field terpisah, bukan tag `<think>` | Output `Reasoning` selalu kosong |
| `reasoning_effort` (`low/medium/high/xhigh`, tanpa `none`) | Tidak diekspos sama sekali |
| `max_completion_tokens` vs `max_tokens` | Node itu hanya kirim `max_tokens` |
| Vision `image_url` + data-URI (jpg/png, maks 20 MiB) | Didukung, aman |
| `seed`, `temperature`, `top_p` | Didukung, aman |

Jadi node Grokified ini dibuat supaya benar-benar pas: seed selalu > 0, reasoning
diambil dari `reasoning_content`, `reasoning_effort` bisa dipilih, dan parameter yang
ditolak server otomatis di-drop lalu diulang (lihat `_downgrade()` di `grokified_client.py`).

## Isi paket

```
ComfyUI-Grokified-API/
├─ __init__.py            # registrasi node
├─ grokified_client.py    # HTTP client (requests saja, tanpa paket openai)
├─ nodes.py               # node + preset system prompt + encoding gambar
└─ tests/self_test.py     # self-test end-to-end ke API asli
```

## Instalasi

```powershell
# dari folder ini
Copy-Item . 'C:\ComfyUI\custom_nodes\ComfyUI-Grokified-API' -Recurse -Force
# lalu restart ComfyUI
```

Tidak ada dependensi baru: `requests` sudah ada di venv ComfyUI.

## API key

Urutan pencarian:

1. widget `api_key` pada node (kosongkan kalau mau pakai env var), lalu
2. environment variable `GROKIFIED_API_KEY`.

Key tidak pernah ditulis penuh ke log — console hanya menampilkan bentuk tersamarkan
(mis. `gk_live_ab...yz`).

## Node: `LLM Text Processor (Grokified API)`

Kategori `LLM/Grokified`.

**Input**

| Nama | Tipe | Catatan |
|---|---|---|
| `model` | COMBO | Diambil live dari `GET /v1/models` (cache 5 menit, ada fallback offline) |
| `system_prompt` | COMBO | Preset `.txt` dari `ComfyUI/models/LLM/prompts` — folder yang sama dengan node lokal |
| `prompt` | STRING | Prompt user (bisa disambung dari node lain) |
| `max_tokens` | INT | Dikirim sebagai `max_completion_tokens` |
| `temperature`, `top_p` | FLOAT | Sampling |
| `reasoning_effort` | COMBO | `default` / `low` / `medium` / `high` / `xhigh` (`default` = field tidak dikirim) |
| `seed` | INT | **min 1** — Grokified menolak `seed = 0` |
| `timeout_seconds` | INT | Batas tunggu request |
| `image` | IMAGE | Opsional; batch didukung, dibatasi `max_images` |
| `system_prompt_text` | STRING | Prompt sistem inline; menang atas preset |
| `api_key`, `base_url` | STRING | Advanced; `base_url` bisa diarahkan ke gateway OpenAI-compatible lain |
| `image_detail`, `max_image_side`, `max_images` | — | Advanced; kontrol pengiriman gambar (auto-downscale, PNG→JPEG bila > 18 MiB) |
| `stream` | BOOLEAN | Advanced; default `True` (progress bar + tahan request panjang) |
| `extra_body_json` | STRING | Advanced; JSON tambahan yang di-merge ke body request |
| `enable_processing` | BOOLEAN | Advanced; `False` = teruskan `prompt` ke `RESPONSE` tanpa memanggil API |

**Output**

| Slot | Nama | Isi |
|---|---|---|
| 0 | `RESPONSE` | Jawaban final (tanpa reasoning) |
| 1 | `REASONING` | `reasoning_content` dari model |
| 2 | `PERF` | `model \| token in/out (reasoning) \| detik \| biaya \| saldo` |
| 3 | `USAGE_JSON` | Metadata usage mentah (untuk logging/cost tracking) |

## Perbedaan dengan node lokal `LLM Text Processor`

- Tidak ada `mmproj`, `ctx_size`, `n_gpu_layers`, `top_k`, `repeat_penalty`,
  `memory_mode`, `extra_args` — itu semua parameter llama.cpp, tidak berlaku untuk API.
- `reasoning` (`auto/on/off`) → `reasoning_effort` (`default/low/medium/high/xhigh`).
  Tidak ada level "off" di Grokified; `low` adalah yang paling murah.
- Preset `system_prompt` yang filenya sudah hilang **tidak** membuat node gagal:
  node memberi warning di console dan lanjut tanpa system prompt. Isi teksnya di
  `system_prompt_text` kalau itu terjadi.

## Self-test

```powershell
$env:GROKIFIED_API_KEY = [Environment]::GetEnvironmentVariable('GROKIFIED_API_KEY','User')
C:\ComfyUI\venv\Scripts\python.exe tests\self_test.py
```

Menguji: daftar model live, output slot, panggilan teks (stream & non-stream),
vision, pembatasan batch, preset yang hilang, mode passthrough, JSON invalid, dan
API key salah. Terakhir dijalankan: **15/15 lulus**.

## Memakai di workflow yang sudah ada

`patch_workflow_to_grokified.py` (ada di folder induk) mengganti node `LLMTextProcessor`
menjadi node ini, memetakan link berdasarkan nama, dan membawa nilai widget yang sama:

```powershell
python patch_workflow_to_grokified.py "workflow.json" "workflow (grokified).json"
```
