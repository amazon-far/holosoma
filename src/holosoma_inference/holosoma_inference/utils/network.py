"""Network interface auto-detection for robot communication."""

import os

_SKIP_PREFIXES = ("lo", "wl", "docker", "br-", "veth", "virbr", "vnet", "tun", "tap")


def detect_robot_interface() -> str:
    """Return the name of the single wired NIC that is operationally UP."""
    for ifname in sorted(os.listdir("/sys/class/net/")):
        if any(ifname.startswith(p) for p in _SKIP_PREFIXES):
            continue
        try:
            with open(f"/sys/class/net/{ifname}/operstate") as f:
                if f.read().strip().lower() != "up":
                    continue
        except OSError:
            continue
        print(f"[network] auto-detected interface: {ifname}")
        return ifname
    print("[network] no wired NIC found, falling back to loopback (lo)")
    return "lo"
