# Vulkan loader build inputs

The Linux bundle uses the Vulkan loader and matching headers from the upstream
`vulkan-sdk-1.4.357.0` tags:

- https://github.com/KhronosGroup/Vulkan-Loader/tree/vulkan-sdk-1.4.357.0
- https://github.com/KhronosGroup/Vulkan-Headers/tree/vulkan-sdk-1.4.357.0

Apply the three patches in this directory to a separate loader checkout, using
`patch -p1` from that checkout. They preserve duplicate-driver detection by
loaded library handle, prioritize matching Vulkan headers and disable an unused
macOS framework install rule. They contain no prebuilt library.

Build/install the headers into a staging directory, then configure the loader
with that headers prefix. Set `CMAKE_NO_SYSTEM_FROM_IMPORTED=ON`,
`CMAKE_INSTALL_PREFIX=/usr`, `CMAKE_INSTALL_SYSCONFDIR=/etc`, `SYSCONFDIR=/etc`
and `FALLBACK_DATA_DIRS=/usr/local/share:/usr/share`. Build tests are disabled;
desktop WSI support remains enabled. Build with the Linux release glibc target
and baseline-compatible X11/Wayland development dependencies. Strip debug
symbols from the final library before packaging.

`zigcc231` and `zigcxx231` invoke Zig for the release C/C++ target; put this
directory on PATH when selecting them as compilers. See
[release-builds.md](../../../docs/release-builds.md) for toolchain versions,
OpenSSL settings, runtime headers and bundle layout.

The Vulkan upstream source retains its own licenses and attribution. Preserve
those notices when redistributing the loader or its source.
