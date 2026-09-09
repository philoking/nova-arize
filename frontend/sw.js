// Service worker for a same-origin LAN app. It exists so Nova is installable and
// can launch offline, but it must NEVER serve a stale app: a deploy has to show
// up immediately with no hard-refresh. So the shell is fetched network-first with
// the HTTP cache bypassed, and the cached copy is only an offline fallback.

const CACHE = "nova-voice-v17";
const SHELL = [
  "/",
  "/index.html",
  "/styles.css",
  "/app.js",
  "/manifest.webmanifest",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
  // hls.js (vendored) — the live-camera player on browsers without native HLS.
  "/vendor/hls.light.min.js",
  // Self-hosted design fonts — precached so the constellation UI renders in its
  // intended type offline, with no external font request.
  "/fonts/fonts.css",
  "/fonts/rajdhani-500.woff2",
  "/fonts/rajdhani-600.woff2",
  "/fonts/rajdhani-700.woff2",
  "/fonts/ibm-plex-mono-400.woff2",
  "/fonts/ibm-plex-mono-500.woff2",
  "/fonts/ibm-plex-sans.woff2",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Never intercept the API — voice/chat must hit the live backend.
  if (url.pathname.startsWith("/api/")) return;
  if (request.method !== "GET") return;

  // Always fetch fresh from the network with the HTTP cache bypassed, refresh the
  // offline copy, and fall back to the cache only when the network is down.
  event.respondWith(
    fetch(request, { cache: "no-store" }).then((resp) => {
      if (resp.ok && url.origin === location.origin) {
        const copy = resp.clone();
        caches.open(CACHE).then((c) => c.put(request, copy));
      }
      return resp;
    }).catch(() => caches.match(request))
  );
});
