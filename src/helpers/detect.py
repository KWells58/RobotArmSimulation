import openvr
import sys
import time

def main():
    print("[DEBUG] starting...", flush=True)

    openvr.init(openvr.VRApplication_Scene)
    print("[DEBUG] openvr.init OK", flush=True)

    vr = openvr.VRSystem()
    print("[DEBUG] VRSystem OK", flush=True)

    # Wait a moment so SteamVR populates devices
    time.sleep(0.2)

    print("\n[OPENVR] Tracked devices:", flush=True)
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        cls = vr.getTrackedDeviceClass(i)
        if cls == 0:
            continue
        role = None
        try:
            role = vr.getControllerRoleForTrackedDeviceIndex(i)
        except Exception:
            pass
        print(f"  id={i:2d} class={cls} role={role}", flush=True)

    openvr.shutdown()
    print("[DEBUG] shutdown OK", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", repr(e), flush=True)
        raise
