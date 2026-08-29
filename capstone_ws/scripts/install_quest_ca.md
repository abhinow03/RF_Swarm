# Sideloading the mkcert root CA onto the Quest (one-time, for zero-warning wss)

The rosbridge wss cert is signed by a **project-contained** mkcert CA at
`capstone_ws/certs/mkcert-ca/rootCA.pem`. We deliberately did NOT run
`mkcert -install`, so **nothing in mars-4090's system/user trust store was
modified** (shared-machine safe). Trust is established on the CLIENT instead:
the Quest must trust this one CA file, after which `wss://` connects with no
warning and no per-connection accept step.

**This step requires the physical headset — it could not be done from the build
machine.** Procedure + the Quest-specific gotchas below.

## Transfer the CA to the Quest
`rootCA.pem` must reach the headset. Easiest options:
- **adb** (Quest in developer mode, USB): `adb push capstone_ws/certs/mkcert-ca/rootCA.pem /sdcard/Download/rootCA.crt`
  (note the `.crt` extension — Android's cert installer filters by it; the file
  content is unchanged PEM).
- Or serve it on the LAN and download it in the Quest Browser, then rename to `.crt`.

## Install as a trusted CA (Horizon OS is Android-based)
Settings path varies by Horizon OS version; the underlying Android flow is:
**Settings → Security (or Privacy) → Encryption & credentials → Install a
certificate → CA certificate → select `rootCA.crt`.**
Android shows *"Your network may be monitored by an unknown third party"* — that
warning is EXPECTED for any user-installed CA and is benign here (it's our own CA).

## FLAGGED Quest-specific friction (verify on the real device)
1. **Horizon OS may not surface the standard Android "Install a certificate"
   menu.** Meta's UI hides parts of Android Settings. If the menu is absent, the
   install may need to be driven via `adb` / a settings intent, or through the
   file manager's "open certificate" action. This is the most likely snag — try
   the Settings path first, fall back to adb.
2. **Browser vs app trust.** WebXR runs in the Quest **Browser** (Chromium). On
   Android, Chromium honors *user-installed* CAs for browser https/wss (unlike
   native apps, which ignore user CAs unless they opt in). So the browser path is
   the one that matters and should work — but CONFIRM by loading
   `https://mars-4090:9443` (or the LAN/tailscale IP) in the Quest Browser and
   checking there is no cert warning before wiring the client.
3. **Address must be in the cert SAN.** The Quest must connect to one of:
   `mars-4090`, `localhost`, `192.168.31.141` (LAN), `100.107.251.96` (tailscale),
   `127.0.0.1`. All are already in the leaf cert — pick whichever the Quest can
   route to on the demo network.

## Fallback (only if the Quest refuses user-CA browser trust)
Revert `run_rosbridge_wss.sh` to the self-signed cert and accept-once per device
(`certs/rosbridge_cert.pem` still present). This reintroduces the manual per-device
step mkcert was chosen to avoid — so treat it as a last resort and flag it.

## Verify (from any machine that trusts the CA — already proven on the build host)
`ssl.create_default_context(cafile=certs/mkcert-ca/rootCA.pem)` + connect
`wss://localhost:9443` verifies cleanly (hostname check ON) — see the phase-6
three-topic test.
