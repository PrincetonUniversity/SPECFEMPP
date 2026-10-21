#pragma once

#include "specfem/plugin/injection_abi.h"
#include <cstdint>
#include <memory>
#include <string>

namespace specfem {
namespace injection {

/**
 * @brief Prefix of a plugin specification that selects a statically
 * registered plugin instead of a shared library.
 */
inline constexpr const char *static_plugin_prefix = "static:";

/**
 * @brief An instance of an injection plugin, created by
 * specfem::injection::PluginLibrary::create.
 *
 * Owns the plugin-side state and destroys it on destruction. The instance
 * keeps the plugin library loaded for as long as it lives.
 */
class PluginInstance {
public:
  PluginInstance(const PluginInstance &) = delete;
  PluginInstance &operator=(const PluginInstance &) = delete;
  PluginInstance(PluginInstance &&other) noexcept;
  PluginInstance &operator=(PluginInstance &&other) noexcept;
  ~PluginInstance();

  /**
   * @brief Plugin-side state, to pass to the other table functions.
   * @return The opaque handle returned by the plugin's @c create
   */
  void *get() const { return self_; }

  /**
   * @brief Function table of the plugin that created this instance.
   * @return The plugin's function table
   */
  const spfp_injection_plugin &table() const { return *table_; }

  /**
   * @brief Last error reported by this instance.
   * @return The plugin's error message, or an empty string
   */
  std::string last_error() const;

private:
  friend class PluginLibrary;
  PluginInstance(std::shared_ptr<void> library,
                 const spfp_injection_plugin *table, void *self);

  void release();

  std::shared_ptr<void> library_;                ///< Keeps the library loaded
  const spfp_injection_plugin *table_ = nullptr; ///< Plugin function table
  void *self_ = nullptr;                         ///< Plugin-side state
};

/**
 * @brief An injection plugin, verified and loaded.
 *
 * A plugin is either a shared library given by its absolute path, or a
 * function table registered in-process with
 * specfem::injection::register_static_plugin. Before a shared library is
 * loaded, its path, ownership, permissions, and embedded manifest are checked,
 * so a plugin that fails those checks never runs code in this process.
 *
 * @code
 * const auto plugin =
 *     specfem::injection::PluginLibrary::open("/opt/plugins/libpw.so");
 * auto instance = plugin.create(config_path, info, communicator);
 * @endcode
 */
class PluginLibrary {
public:
  /**
   * @brief Open a plugin from its specification.
   *
   * @param spec Absolute path to a shared library, or @c "static:<name>" for a
   * statically registered plugin
   * @return The loaded plugin
   * @throws std::runtime_error if the plugin cannot be found, fails a check,
   * or cannot be loaded
   */
  static PluginLibrary open(const std::string &spec);

  /**
   * @brief Verify and load a plugin shared library.
   *
   * The path must be absolute and resolve to a regular file. The file and its
   * directory must be owned by the current user or root and must not be
   * group- or world-writable. The file must embed exactly one manifest for
   * this ABI version. Only then is it loaded with @c dlopen.
   *
   * @param path Absolute path to the shared library
   * @return The loaded plugin
   * @throws std::runtime_error if a check fails or the library cannot be
   * loaded
   */
  static PluginLibrary open_file(const std::string &path);

  /**
   * @brief Look up a statically registered plugin.
   *
   * @param name Name the plugin was registered under
   * @return The plugin
   * @throws std::runtime_error if no plugin is registered under @p name
   */
  static PluginLibrary open_static(const std::string &name);

  /**
   * @brief The plugin's function table.
   * @return Function table, valid while this object (or an instance it
   * created) is alive
   */
  const spfp_injection_plugin &table() const { return *table_; }

  /**
   * @brief Plugin name, as reported by the plugin.
   * @return Plugin name
   */
  std::string name() const { return table_->name; }

  /**
   * @brief Plugin version string, as reported by the plugin.
   * @return Plugin version
   */
  std::string version() const { return table_->version; }

  /**
   * @brief ABI version the plugin was built against.
   * @return ABI version
   */
  std::uint32_t abi_version() const { return table_->abi_version; }

  /**
   * @brief Where the plugin came from.
   * @return Resolved path of the shared library, or @c "static:<name>"
   */
  const std::string &source() const { return source_; }

  /**
   * @brief Create a plugin instance.
   *
   * @param config_path Plugin configuration file, passed to the plugin as is
   * @param info Simulation parameters
   * @param communicator Communicator of the ranks taking part in the run
   * @return The new instance
   * @throws std::runtime_error with the plugin's error message if @c create
   * fails
   */
  PluginInstance create(const std::string &config_path,
                        const spfp_simulation_info &info,
                        const spfp_communicator &communicator) const;

private:
  PluginLibrary(std::shared_ptr<void> handle,
                const spfp_injection_plugin *table, std::string source);

  std::shared_ptr<void> handle_; ///< dlopen handle; null for static plugins
  const spfp_injection_plugin *table_ = nullptr; ///< Plugin function table
  std::string source_; ///< Resolved path or "static:<name>"
};

/**
 * @brief Register a plugin function table in-process, without a shared
 * library.
 *
 * The plugin can then be opened as @c "static:<name>", where @c name is the
 * table's name. Used by tests and by builds that link plugins statically.
 *
 * @param table Function table; must outlive every use of the plugin
 * @throws std::runtime_error if the table is invalid or a plugin with the same
 * name is already registered
 */
void register_static_plugin(const spfp_injection_plugin *table);

} // namespace injection
} // namespace specfem
