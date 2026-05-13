# Sim-to-Real 실행 메뉴얼

`sim_to_real_phuma.sh` 를 실행해서 실제 G1 로봇에서 정책을 돌리기 위한 절차.

## 0. 환경 설치 (최초 1 회만)

`hsinference` conda env 가 없으면 먼저 설치:

```bash
bash scripts/setup_inference.sh
```


## 1. 사전 준비

### 네트워크 (이더넷)

- 로봇과 PC 를 이더넷으로 연결.
- `ip addr` 로 로봇이 물려있는 인터페이스 이름 확인 (예: `eth0`, `enp3s0` 등), 
- `sim_to_real_phuma.sh` 의 `--task.interface` 값에 그대로 적기.

### 모델 / 모션 파일

스크립트 안에 다음 두 경로가 실제로 존재하는지 확인:
- `--task.model-path` : 실행할 ONNX policy
- `--task.motion-file-path` : 재생할 모션 npz

다른 모션을 돌리고 싶으면 `--task.motion-file-path` 만 다른 npz 로 바꾸면 됨.

### 조이스틱

- 조이스틱(예: G1 컨트롤러) PC 에 연결.
- 키보드로 돌리고 싶으면 스크립트에서 `--task.use-joystick` 줄을 빼면 됨.

## 2. 실행 절차

```bash
bash scripts/sim_to_real/sim_to_real_phuma.sh
```

1. 정책 / 모션 / 로봇이 로드되면서 controls 가이드가 터미널에 출력됨.
2. **`⚠️ Ready to enter stiff hold mode` → `Press Enter to continue...`** 가 뜨면
   엔터 키를 눌러야 stiff hold 모드로 진입함 (모터에 힘이 들어감).
3. 이 시점부터 조이스틱 / 키보드로 제어 가능.

## 3. 컨트롤 (조이스틱)

순서대로 누르는 게 일반적인 워크플로우.

| 버튼 | 동작 |
|------|------|
| **A** | 정책 시작 (policy actions 사용 시작) |
| **Start** | 모션 클립 재생 시작 (WBT 모드) |
| **B** | 정책 중지 → 다시 stiff hold 로 복귀 |
| **L1 + R1** | 컨트롤러 프로그램 종료 (kill) |

**전형적인 시퀀스:**
1. 키보드 `Enter` 눌러서 준비.
2. `A` 로 정책 가동.
3. `Start` 로 모션 재생.
4. 모션이 끝나면 자동으로 stiff hold 로 돌아감. 다시 재생하려면 `Start`.
5. 끝나면 `B` 로 정책 중지 → `L1+R1` 로 종료.

## 4. 컨트롤 (키보드 — `--task.use-joystick` 뺐을 때)

터미널이 active 상태여야 입력 받음.

| 키 | 동작 |
|---|------|
| `i` | default pose 로 |
| `]` | 정책 시작 |
| `s` | 모션 클립 재생 시작 |
| `o` | 정책 중지 |
| `v` / `b` | KP -0.01 / +0.01 |
| `f` / `g` | KP -0.1 / +0.1 |
| `r` | KP 를 1.0 으로 리셋 |

## 5. 트러블슈팅

- **로봇 상태가 안 들어옴**: 이더넷 케이블 / 인터페이스 이름 / UFW 다시 확인.
- **`⚠️ Non-interactive mode detected`**: 터미널이 TTY 가 아닐 때 발생. stiff hold confirm 프롬프트가 스킵됨 — 직접 터미널에서 실행할 것.
- **모션이 끝났는데 다시 안 됨**: 자동으로 stiff hold 로 갔으니 `Start` (조이스틱) 또는 `s` (키보드) 로 다시 트리거.
