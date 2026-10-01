import numpy as np
import matplotlib.pyplot as plt

# Import directly from your ttv_core module
from ttv_core import ttv_rl_episode, example_path, TTVConfig


def main():
    # 1. Generate reference path (0 to 150m double lane change)
    x = np.linspace(0.0, 150.0, 751)
    y = 1.75 * (np.tanh(0.12 * (x - 35.0)) - np.tanh(0.12 * (x - 80.0)))
    path_input = np.column_stack((x, y))

    # 2. Speed Sweep Test across various speeds [m/s]
    v_test = [8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 25.0]
    results = []

    print("\n=== TTV RL Episode Speed Sweep Test ===")
    print(
        f"{'Speed (m/s)':<11} | {'Status':<8} | {'Max Slip (rad)':<14} | {'RMS Lat Err':<14} | {'Failure Reason':<25}"
    )
    print("-" * 85)

    for v in v_test:
        cfg = TTVConfig(V=v, logPhysicalDiagnostics=True)

        try:
            reward, rewards, log = ttv_rl_episode(path_input, cfg)

            status_str = "FAIL" if rewards.get("failed", False) else "PASS"
            failure_reason = rewards.get("failureReason", "None")
            max_slip = rewards.get("maxSlipFront", 0.0)
            rms_err = log["metrics"].get("rmsLateralError", 0.0)

            results.append(
                {
                    "speed": v,
                    "passed": rewards.get("passed", False),
                    "reason": failure_reason,
                    "max_slip": max_slip,
                    "rms_err": rms_err,
                    "plant_state": log.get("plantState", None),
                }
            )

            print(
                f"{v:<11.1f} | {status_str:<8} | {max_slip:<14.4f} | {rms_err:<14.4f} | {failure_reason:<25}"
            )

        except Exception as e:
            print(f"{v:<11.1f} | {'ERROR':<8} | {'N/A':<14} | {'N/A':<14} | {str(e):<25}")

    # 3. Plotting Trajectories & Slip Limits
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Trajectory Plot
    ax1.plot(x, y, "k--", linewidth=2, label="Reference Path")
    ax1.set_ylabel("Y [m]")
    ax1.set_xlabel("X [m]")
    ax1.set_title("Double Lane Change Trajectories Across Speeds")
    ax1.grid(True)

    for res in results:
        if res.get("plant_state") is not None:
            state = res["plant_state"]
            status = "PASS" if res["passed"] else "FAIL"
            # Plant state: X1 is row 0, Y1 is row 1
            ax1.plot(
                state[0, :],
                state[1, :],
                label=f"V = {res['speed']} m/s ({status})",
            )
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Tire Slip Bar Chart
    speeds = [r["speed"] for r in results]
    slips = [r["max_slip"] for r in results]

    ax2.bar(speeds, slips, width=1.5, color="skyblue", edgecolor="black")
    ax2.axhline(
        0.5,
        color="r",
        linestyle="--",
        linewidth=1.5,
        label="Slip Limit Threshold (0.2 rad)",
    )
    ax2.set_xlabel("Vehicle Speed V [m/s]")
    ax2.set_ylabel("Max Front Slip Angle [rad]")
    ax2.set_title("Peak Tire Slip Angle vs Speed")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()