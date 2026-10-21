#include "specfem/injection/plugin_library.hpp"
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace specfem::injection_impl {

/** @brief Plugins registered with register_static_plugin, by name. */
std::map<std::string, const spfp_injection_plugin *> &static_registry();

/** @brief Throw unless @p table is a complete table for this ABI version. */
void validate_table(const spfp_injection_plugin *table,
                    const std::string &source);

/** @brief Throw unless @p path and its directory are safe to load from. */
void check_permissions(const std::string &path);

/** @brief Read and validate the single manifest embedded in @p path. */
spfp_manifest read_manifest(const std::string &path);

} // namespace specfem::injection_impl

std::map<std::string, const spfp_injection_plugin *> &
specfem::injection_impl::static_registry() {
  static std::map<std::string, const spfp_injection_plugin *> registry;
  return registry;
}

void specfem::injection_impl::validate_table(const spfp_injection_plugin *table,
                                             const std::string &source) {
  const auto fail = [&](const std::string &reason) {
    throw std::runtime_error("Injection plugin " + source +
                             " has an invalid function table: " + reason);
  };
  if (table == nullptr)
    fail("the entry point returned a null table.");
  if (table->abi_version != SPFP_INJECTION_ABI_VERSION)
    fail("it was built for ABI version " + std::to_string(table->abi_version) +
         ", but SPECFEM++ supports version " +
         std::to_string(SPFP_INJECTION_ABI_VERSION) +
         ". Rebuild the plugin against a matching plugin SDK.");
  if (table->reserved != 0)
    fail("the reserved field is not 0.");
  if (table->name == nullptr || table->name[0] == '\0')
    fail("the plugin name is empty.");
  if (table->version == nullptr)
    fail("the plugin version is null.");
  if (table->create == nullptr || table->supports == nullptr ||
      table->setup == nullptr || table->fill == nullptr ||
      table->last_error == nullptr || table->destroy == nullptr)
    fail("a required function pointer is null.");
}

void specfem::injection_impl::check_permissions(const std::string &path) {
  const auto check = [](const std::string &target, const std::string &what) {
    struct stat status{};
    if (::stat(target.c_str(), &status) != 0)
      throw std::runtime_error("Cannot inspect injection plugin " + what +
                               " '" + target + "': " + std::strerror(errno));
    if (status.st_uid != ::geteuid() && status.st_uid != 0)
      throw std::runtime_error(
          "Refusing to load injection plugin: " + what + " '" + target +
          "' is owned by another user (uid " + std::to_string(status.st_uid) +
          "). It must be owned by you or by root.");
    if ((status.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
      std::ostringstream message;
      message << "Refusing to load injection plugin: " << what << " '" << target
              << "' is group- or world-writable (mode " << std::oct
              << (status.st_mode & 07777) << "). Run 'chmod go-w " << target
              << "' to fix it.";
      throw std::runtime_error(message.str());
    }
  };
  check(path, "file");
  check(std::filesystem::path(path).parent_path().string(), "directory");
}

spfp_manifest specfem::injection_impl::read_manifest(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Cannot read injection plugin '" + path + "'.");
  const std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());

  const char magic[16] = SPFP_MANIFEST_MAGIC_INITIALIZER;
  spfp_manifest manifest{};
  int count = 0;
  for (auto it = std::search(bytes.begin(), bytes.end(), std::begin(magic),
                             std::end(magic));
       it != bytes.end();
       it = std::search(it + 1, bytes.end(), std::begin(magic),
                        std::end(magic))) {
    if (static_cast<std::size_t>(bytes.end() - it) < sizeof(spfp_manifest))
      throw std::runtime_error("Injection plugin '" + path +
                               "' has a truncated manifest.");
    std::memcpy(&manifest, &*it, sizeof(spfp_manifest));
    ++count;
  }

  const auto fail = [&](const std::string &reason) {
    throw std::runtime_error("Refusing to load injection plugin '" + path +
                             "': " + reason);
  };
  if (count == 0)
    fail("it has no plugin manifest. Build it with "
         "specfempp_add_injection_plugin() and declare it with "
         "SPECFEMPP_DEFINE_INJECTION_PLUGIN or "
         "SPFP_DEFINE_INJECTION_MANIFEST.");
  if (count > 1)
    fail("it has " + std::to_string(count) +
         " plugin manifests; a plugin must define exactly one.");
  if (manifest.manifest_version != SPFP_MANIFEST_VERSION)
    fail("unsupported manifest version " +
         std::to_string(manifest.manifest_version) + " (expected " +
         std::to_string(SPFP_MANIFEST_VERSION) + ").");
  if (manifest.plugin_kind != SPFP_PLUGIN_KIND_INJECTION)
    fail("it is not an injection plugin (plugin kind " +
         std::to_string(manifest.plugin_kind) + ").");
  if (manifest.abi_version != SPFP_INJECTION_ABI_VERSION)
    fail("it was built for ABI version " +
         std::to_string(manifest.abi_version) +
         ", but SPECFEM++ supports version " +
         std::to_string(SPFP_INJECTION_ABI_VERSION) +
         ". Rebuild the plugin against a matching plugin SDK.");
  if (std::memchr(manifest.name, '\0', sizeof(manifest.name)) == nullptr ||
      std::memchr(manifest.version, '\0', sizeof(manifest.version)) == nullptr)
    fail("its manifest name or version is not NUL-terminated.");
  return manifest;
}

specfem::injection::PluginInstance::PluginInstance(
    std::shared_ptr<void> library, const spfp_injection_plugin *table,
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
    std::shared_ptr<void> handle, const spfp_injection_plugin *table,
    std::string source)
    : handle_(std::move(handle)), table_(table), source_(std::move(source)) {}

specfem::injection::PluginLibrary
specfem::injection::PluginLibrary::open(const std::string &spec) {
  const std::string prefix = specfem::injection::static_plugin_prefix;
  if (spec.compare(0, prefix.size(), prefix) == 0)
    return open_static(spec.substr(prefix.size()));
  return open_file(spec);
}

specfem::injection::PluginLibrary
specfem::injection::PluginLibrary::open_file(const std::string &path) {
  if (!std::filesystem::path(path).is_absolute())
    throw std::runtime_error(
        "Injection plugin '" + path +
        "' is not an absolute path. Pass an absolute path, or " +
        std::string(specfem::injection::static_plugin_prefix) +
        "<name> for a statically registered plugin; library search paths are "
        "never used.");

  char resolved_buffer[PATH_MAX];
  if (::realpath(path.c_str(), resolved_buffer) == nullptr)
    throw std::runtime_error("Cannot resolve injection plugin '" + path +
                             "': " + std::strerror(errno));
  const std::string resolved(resolved_buffer);
  if (!std::filesystem::is_regular_file(resolved))
    throw std::runtime_error("Injection plugin '" + path + "' (resolved to '" +
                             resolved + "') is not a regular file.");

  specfem::injection_impl::check_permissions(resolved);
  const spfp_manifest manifest =
      specfem::injection_impl::read_manifest(resolved);

  void *raw_handle = ::dlopen(resolved.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (raw_handle == nullptr) {
    const char *error = ::dlerror();
    throw std::runtime_error("Cannot load injection plugin '" + resolved +
                             "': " + (error ? error : "unknown dlopen error"));
  }
  std::shared_ptr<void> handle(raw_handle,
                               [](void *handle) { ::dlclose(handle); });

  const auto entry = reinterpret_cast<spfp_injection_plugin_entry>(
      ::dlsym(handle.get(), SPFP_INJECTION_PLUGIN_ENTRY_NAME));
  if (entry == nullptr)
    throw std::runtime_error(
        "Injection plugin '" + resolved + "' does not export " +
        SPFP_INJECTION_PLUGIN_ENTRY_NAME +
        "(). Was it built with specfempp_add_injection_plugin() and "
        "SPECFEMPP_DEFINE_INJECTION_PLUGIN?");

  const spfp_injection_plugin *table = entry();
  specfem::injection_impl::validate_table(table, "'" + resolved + "'");
  if (std::string(table->name) != manifest.name)
    throw std::runtime_error(
        "Injection plugin '" + resolved + "' reports the name '" + table->name +
        "', but its manifest says '" + manifest.name + "'.");

  return PluginLibrary(std::move(handle), table, resolved);
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
                       specfem::injection::static_plugin_prefix + name);
}

specfem::injection::PluginInstance specfem::injection::PluginLibrary::create(
    const std::string &config_path, const spfp_simulation_info &info,
    const spfp_communicator &communicator) const {
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
    const spfp_injection_plugin *table) {
  specfem::injection_impl::validate_table(table, "(static)");
  auto &registry = specfem::injection_impl::static_registry();
  if (!registry.emplace(table->name, table).second)
    throw std::runtime_error("An injection plugin named '" +
                             std::string(table->name) +
                             "' is already statically registered.");
}
