#include "specfem/injection/plugin_library.hpp"
#include "specfem/utilities/sha256.hpp"
// Shared-library plugins need POSIX dlopen and file modes; on Windows only
// statically registered plugins are available.
#if !defined(_WIN32)
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace specfem::injection_impl {

/** @brief Plugins registered with register_static_plugin, by name. */
std::map<std::string, const specfempp_injection_table *> &static_registry();

/** @brief Throw unless @p table is a complete table for this ABI version. */
void validate_table(const specfempp_injection_table *table,
                    const std::string &source);

/** @brief Error for a shared-library plugin on a platform without them. */
[[noreturn]] void throw_unsupported_platform(const std::string &path);

/** @brief Read the whole of @p path. */
std::vector<char> read_file(const std::string &path);

/** @brief Find and validate the single manifest in @p bytes, read from
 * @p path. */
specfempp_manifest read_manifest(std::span<const char> bytes,
                                 const std::string &path);

} // namespace specfem::injection_impl

std::map<std::string, const specfempp_injection_table *> &
specfem::injection_impl::static_registry() {
  static std::map<std::string, const specfempp_injection_table *> registry;
  return registry;
}

void specfem::injection_impl::validate_table(
    const specfempp_injection_table *table, const std::string &source) {
  const auto fail = [&](const std::string &reason) {
    throw std::runtime_error("Injection plugin " + source +
                             " has an invalid function table: " + reason);
  };
  if (table == nullptr)
    fail("the entry point returned a null table.");
  if (table->abi_version != SPECFEMPP_INJECTION_ABI_VERSION)
    fail("it was built for ABI version " + std::to_string(table->abi_version) +
         ", but SPECFEM++ supports version " +
         std::to_string(SPECFEMPP_INJECTION_ABI_VERSION) +
         ". Rebuild the plugin against a matching plugin SDK.");
  if (table->reserved != 0)
    fail("the reserved field is not 0.");
  if (table->name == nullptr || table->name[0] == '\0')
    fail("the plugin name is empty.");
  if (table->version == nullptr)
    fail("the plugin version is null.");
  const std::array has_function = {
    table->create != nullptr,     table->supports != nullptr,
    table->setup != nullptr,      table->fill != nullptr,
    table->last_error != nullptr, table->destroy != nullptr
  };
  if (!std::ranges::all_of(has_function, std::identity{}))
    fail("a required function pointer is null.");
}

void specfem::injection_impl::throw_unsupported_platform(
    const std::string &path) {
  throw std::runtime_error(
      "Cannot load injection plugin '" + path +
      "': shared-library injection plugins are supported on Linux and macOS "
      "only. Use " +
      std::string(specfem::injection::static_plugin_prefix) +
      "<name> for a statically registered plugin.");
}

#if defined(_WIN32)

bool specfem::injection_impl::running_privileged() { return false; }

void specfem::injection_impl::check_permissions(
    const std::string &path, [[maybe_unused]] const bool privileged) {
  specfem::injection_impl::throw_unsupported_platform(path);
}

#else

bool specfem::injection_impl::running_privileged() {
  return ::geteuid() == 0 || ::getuid() != ::geteuid();
}

void specfem::injection_impl::check_permissions(const std::string &path,
                                                const bool privileged) {
  const auto status_of = [](const std::filesystem::path &target,
                            const std::string &what) {
    struct stat status{};
    if (::stat(target.c_str(), &status) != 0)
      throw std::runtime_error("Cannot inspect injection plugin " + what +
                               " '" + target.string() +
                               "': " + std::strerror(errno));
    return status;
  };
  const auto refuse = [](const std::string &what,
                         const std::filesystem::path &target,
                         const std::string &reason) {
    throw std::runtime_error("Refusing to load injection plugin: " + what +
                             " '" + target.string() + "' " + reason);
  };
  const auto mode_of = [](const struct stat &status) {
    std::ostringstream mode;
    mode << std::oct << (status.st_mode & 07777);
    return mode.str();
  };
  // A sticky directory only lets an entry's owner rename or remove it.
  const auto others_can_replace = [](const struct stat &status,
                                     const mode_t write_bits) {
    return (status.st_mode & write_bits) != 0 &&
           (status.st_mode & S_ISVTX) == 0;
  };

  const std::filesystem::path file(path);
  if (!privileged) {
    // The plugin runs with the user's own rights, so only users outside the
    // group, who could not otherwise run code as this user, are a threat.
    const auto file_status = status_of(file, "file");
    if ((file_status.st_mode & S_IWOTH) != 0)
      refuse("file", file,
             "is world-writable (mode " + mode_of(file_status) +
                 "). Run 'chmod o-w " + path + "' to fix it.");
    const auto directory = file.parent_path();
    const auto directory_status = status_of(directory, "directory");
    if (others_can_replace(directory_status, S_IWOTH))
      refuse("directory", directory,
             "is world-writable without the sticky bit (mode " +
                 mode_of(directory_status) + ").");
    return;
  }

  // Running as root or setuid: anyone who can change the plugin, or any
  // directory above it, gains these elevated rights.
  const auto euid = ::geteuid();
  const auto trusted_owner = [&](const struct stat &status) {
    return status.st_uid == 0 || status.st_uid == euid;
  };
  const auto file_status = status_of(file, "file");
  if (!trusted_owner(file_status))
    refuse("file", file,
           "is owned by uid " + std::to_string(file_status.st_uid) +
               "; a privileged run only loads plugins owned by root or the "
               "effective user.");
  if ((file_status.st_mode & (S_IWGRP | S_IWOTH)) != 0)
    refuse("file", file,
           "is group- or world-writable (mode " + mode_of(file_status) +
               "); a privileged run requires 'chmod go-w'.");
  for (auto directory = file.parent_path();;
       directory = directory.parent_path()) {
    const auto status = status_of(directory, "directory");
    if (!trusted_owner(status))
      refuse("directory", directory,
             "is owned by uid " + std::to_string(status.st_uid) +
                 "; a privileged run requires every directory above the "
                 "plugin to be owned by root or the effective user.");
    if (others_can_replace(status, S_IWGRP | S_IWOTH))
      refuse("directory", directory,
             "is group- or world-writable without the sticky bit (mode " +
                 mode_of(status) +
                 "); a privileged run requires every directory above the "
                 "plugin to be writable only by its owner.");
    if (directory == directory.root_path())
      break;
  }
}

#endif

std::vector<char> specfem::injection_impl::read_file(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Cannot read injection plugin '" + path + "'.");
  return { std::istreambuf_iterator<char>(file),
           std::istreambuf_iterator<char>() };
}

specfempp_manifest
specfem::injection_impl::read_manifest(std::span<const char> bytes,
                                       const std::string &path) {
  const char magic[16] = SPECFEMPP_MANIFEST_MAGIC_INITIALIZER;
  specfempp_manifest manifest{};
  int count = 0;
  // Scan for every occurrence of the magic; the rest of the file after each
  // match must hold a whole manifest.
  for (std::span<const char> rest = bytes;;) {
    const auto match = std::ranges::search(rest, magic);
    if (match.empty())
      break;
    const auto found = rest.subspan(match.begin() - rest.begin());
    if (found.size() < sizeof(specfempp_manifest))
      throw std::runtime_error("Injection plugin '" + path +
                               "' has a truncated manifest.");
    std::memcpy(&manifest, found.data(), sizeof(specfempp_manifest));
    ++count;
    rest = found.subspan(1);
  }

  const auto fail = [&](const std::string &reason) {
    throw std::runtime_error("Refusing to load injection plugin '" + path +
                             "': " + reason);
  };
  if (count == 0)
    fail("it has no plugin manifest. Build it with "
         "specfempp_add_injection_plugin() and declare it with "
         "SPECFEMPP_DEFINE_INJECTION_PLUGIN or "
         "SPECFEMPP_DEFINE_INJECTION_MANIFEST.");
  if (count > 1)
    fail("it has " + std::to_string(count) +
         " plugin manifests; a plugin must define exactly one.");
  if (manifest.manifest_version != SPECFEMPP_MANIFEST_VERSION)
    fail("unsupported manifest version " +
         std::to_string(manifest.manifest_version) + " (expected " +
         std::to_string(SPECFEMPP_MANIFEST_VERSION) + ").");
  if (manifest.plugin_kind != SPECFEMPP_PLUGIN_KIND_INJECTION)
    fail("it is not an injection plugin (plugin kind " +
         std::to_string(manifest.plugin_kind) + ").");
  if (manifest.abi_version != SPECFEMPP_INJECTION_ABI_VERSION)
    fail("it was built for ABI version " +
         std::to_string(manifest.abi_version) +
         ", but SPECFEM++ supports version " +
         std::to_string(SPECFEMPP_INJECTION_ABI_VERSION) +
         ". Rebuild the plugin against a matching plugin SDK.");
  if (std::memchr(manifest.name, '\0', sizeof(manifest.name)) == nullptr ||
      std::memchr(manifest.version, '\0', sizeof(manifest.version)) == nullptr)
    fail("its manifest name or version is not NUL-terminated.");
  return manifest;
}

specfem::injection::PluginInstance::PluginInstance(
    std::shared_ptr<void> library, const specfempp_injection_table *table,
    void *self)
    : library_(std::move(library)), table_(table), self_(self) {}

specfem::injection::PluginInstance::PluginInstance(
    PluginInstance &&other) noexcept
    : library_(std::move(other.library_)),
      table_(std::exchange(other.table_, nullptr)),
      self_(std::exchange(other.self_, nullptr)) {}

specfem::injection::PluginInstance &
specfem::injection::PluginInstance::operator=(PluginInstance &&other) noexcept {
  if (this != &other) {
    release();
    library_ = std::move(other.library_);
    table_ = std::exchange(other.table_, nullptr);
    self_ = std::exchange(other.self_, nullptr);
  }
  return *this;
}

specfem::injection::PluginInstance::~PluginInstance() { release(); }

void specfem::injection::PluginInstance::release() {
  if (table_ != nullptr && self_ != nullptr)
    table_->destroy(self_);
  self_ = nullptr;
}

std::string specfem::injection::PluginInstance::last_error() const {
  const char *message = table_->last_error(self_);
  return message == nullptr ? std::string() : std::string(message);
}

specfem::injection::PluginLibrary::PluginLibrary(
    std::shared_ptr<void> handle, const specfempp_injection_table *table,
    std::string source, std::string sha256)
    : handle_(std::move(handle)), table_(table), source_(std::move(source)),
      sha256_(std::move(sha256)) {}

specfem::injection::PluginLibrary
specfem::injection::PluginLibrary::open(const std::string &spec) {
  constexpr std::string_view prefix = specfem::injection::static_plugin_prefix;
  if (spec.starts_with(prefix))
    return open_static(spec.substr(prefix.size()));
  return open_file(spec);
}

specfem::injection::PluginLibrary
specfem::injection::PluginLibrary::open_file(const std::string &path) {
#if defined(_WIN32)
  specfem::injection_impl::throw_unsupported_platform(path);
#else
  // Only the resolved absolute path reaches dlopen, so library search paths
  // are never used, even for a bare file name.
  char resolved_buffer[PATH_MAX];
  if (::realpath(path.c_str(), resolved_buffer) == nullptr)
    throw std::runtime_error("Cannot resolve injection plugin '" + path +
                             "': " + std::strerror(errno));
  const std::string resolved(resolved_buffer);
  if (!std::filesystem::is_regular_file(resolved))
    throw std::runtime_error("Injection plugin '" + path + "' (resolved to '" +
                             resolved + "') is not a regular file.");

  specfem::injection_impl::check_permissions(
      resolved, specfem::injection_impl::running_privileged());
  const auto bytes = specfem::injection_impl::read_file(resolved);
  const specfempp_manifest manifest =
      specfem::injection_impl::read_manifest(bytes, resolved);
  const auto sha256 =
      specfem::utilities::sha256_hex(std::as_bytes(std::span(bytes)));

  void *raw_handle = ::dlopen(resolved.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (raw_handle == nullptr) {
    const char *error = ::dlerror();
    throw std::runtime_error("Cannot load injection plugin '" + resolved +
                             "': " + (error ? error : "unknown dlopen error"));
  }
  std::shared_ptr<void> handle(raw_handle,
                               [](void *handle) { ::dlclose(handle); });

  const auto entry = reinterpret_cast<specfempp_injection_entry>(
      ::dlsym(handle.get(), SPECFEMPP_INJECTION_PLUGIN_ENTRY_NAME));
  if (entry == nullptr)
    throw std::runtime_error(
        "Injection plugin '" + resolved + "' does not export " +
        SPECFEMPP_INJECTION_PLUGIN_ENTRY_NAME +
        "(). Was it built with specfempp_add_injection_plugin() and "
        "SPECFEMPP_DEFINE_INJECTION_PLUGIN?");

  const specfempp_injection_table *table = entry();
  specfem::injection_impl::validate_table(table, "'" + resolved + "'");
  if (std::string(table->name) != manifest.name)
    throw std::runtime_error(
        "Injection plugin '" + resolved + "' reports the name '" + table->name +
        "', but its manifest says '" + manifest.name + "'.");

  return PluginLibrary(std::move(handle), table, resolved, sha256);
#endif
}

specfem::injection::PluginLibrary
specfem::injection::PluginLibrary::open_static(const std::string &name) {
  const auto &registry = specfem::injection_impl::static_registry();
  const auto found = registry.find(name);
  if (found == registry.end()) {
    std::ostringstream message;
    message << "No statically registered injection plugin named '" << name
            << "'. Registered plugins:";
    if (registry.empty())
      message << " (none)";
    for (const auto &[registered, table] : registry)
      message << " " << registered;
    message << ".";
    throw std::runtime_error(message.str());
  }
  return PluginLibrary(nullptr, found->second,
                       specfem::injection::static_plugin_prefix + name, "");
}

specfem::injection::PluginInstance specfem::injection::PluginLibrary::create(
    const std::string &config_path, const specfempp_simulation_info &info,
    const specfempp_communicator &communicator) const {
  void *self = table_->create(config_path.c_str(), &info, &communicator);
  if (self == nullptr) {
    const char *error = table_->last_error(nullptr);
    throw std::runtime_error(
        "Injection plugin '" + name() + "' failed to create an instance: " +
        (error != nullptr && error[0] != '\0' ? error : "no error message"));
  }
  return PluginInstance(handle_, table_, self);
}

void specfem::injection::register_static_plugin(
    const specfempp_injection_table *table) {
  specfem::injection_impl::validate_table(table, "(static)");
  auto &registry = specfem::injection_impl::static_registry();
  if (!registry.emplace(table->name, table).second)
    throw std::runtime_error("An injection plugin named '" +
                             std::string(table->name) +
                             "' is already statically registered.");
}
