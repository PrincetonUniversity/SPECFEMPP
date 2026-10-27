// Tests for specfem::injection::PluginLibrary: every check performed before
// and after loading a plugin, and the static-registration path.
//
// Plugins are copied into a private temporary directory before loading, so the
// permission checks see known modes regardless of the build tree's umask.

#include "specfem/injection.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>

namespace specfem::injection_loader_test {

// A private (0700) temporary directory, removed with its contents.
class TempDir {
public:
  TempDir() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "specfempp_loader_XXXXXX")
            .string();
    if (::mkdtemp(pattern.data()) == nullptr)
      throw std::runtime_error("mkdtemp failed");
    path_ = pattern;
    std::filesystem::permissions(path_, std::filesystem::perms::owner_all);
  }
  ~TempDir() {
    std::error_code error;
    std::filesystem::permissions(path_, std::filesystem::perms::owner_all,
                                 error);
    std::filesystem::remove_all(path_, error);
  }
  TempDir(const TempDir &) = delete;
  TempDir &operator=(const TempDir &) = delete;

  // Copy a plugin into the directory with mode 0755; returns its path.
  std::string install(const std::string &source,
                      const std::string &name = "plugin.so") const {
    const auto target = path_ / name;
    std::filesystem::copy_file(source, target);
    ::chmod(target.c_str(), 0755);
    return target.string();
  }

  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

// Message of the std::runtime_error thrown by open(spec).
std::string open_error(const std::string &spec) {
  try {
    [[maybe_unused]] const auto plugin =
        specfem::injection::PluginLibrary::open(spec);
  } catch (const std::runtime_error &error) {
    return error.what();
  }
  return "<no error>";
}

specfempp_simulation_info simulation_info() {
  specfempp_simulation_info info{};
  info.dimension = 3;
  info.nstep = 10;
  info.dt = 0.1;
  return info;
}

specfempp_communicator communicator() {
  return specfempp_communicator{ 0, 0, 1, 0 };
}

using ::testing::HasSubstr;

TEST(InjectionLoader, LoadsGoodPlugin) {
  const TempDir dir;
  const auto path = dir.install(SPECFEMPP_TEST_GOOD_PLUGIN);

  const auto plugin = specfem::injection::PluginLibrary::open(path);
  EXPECT_EQ(plugin.name(), "loader-good");
  EXPECT_EQ(plugin.version(), "2.0.1");
  EXPECT_EQ(plugin.abi_version(), SPECFEMPP_INJECTION_ABI_VERSION);
  EXPECT_EQ(plugin.source(), std::filesystem::canonical(path).string());
  EXPECT_EQ(plugin.sha256().size(), 64);

  const auto instance = plugin.create("ok", simulation_info(), communicator());
  EXPECT_NE(instance.get(), nullptr);
  EXPECT_EQ(instance.last_error(), "");
}

TEST(InjectionLoader, InstanceKeepsLibraryLoaded) {
  const TempDir dir;
  const auto path = dir.install(SPECFEMPP_TEST_GOOD_PLUGIN);

  std::optional<specfem::injection::PluginInstance> instance;
  {
    const auto plugin = specfem::injection::PluginLibrary::open(path);
    instance.emplace(plugin.create("ok", simulation_info(), communicator()));
  }
  // The table and destroy must still be callable after the library object
  // is gone.
  EXPECT_EQ(instance->last_error(), "");
  instance.reset();
}

TEST(InjectionLoader, ResolvesSymlinks) {
  const TempDir dir;
  const auto path = dir.install(SPECFEMPP_TEST_GOOD_PLUGIN);
  const auto link = dir.path() / "link.so";
  std::filesystem::create_symlink(path, link);

  const auto plugin = specfem::injection::PluginLibrary::open(link.string());
  EXPECT_EQ(plugin.source(), std::filesystem::canonical(path).string());
}

TEST(InjectionLoader, CreateFailureReportsPluginError) {
  const TempDir dir;
  const auto plugin = specfem::injection::PluginLibrary::open(
      dir.install(SPECFEMPP_TEST_GOOD_PLUGIN));
  try {
    [[maybe_unused]] const auto instance =
        plugin.create("fail-create", simulation_info(), communicator());
    FAIL() << "create should have thrown";
  } catch (const std::runtime_error &error) {
    EXPECT_THAT(error.what(), HasSubstr("'loader-good' failed to create"));
    EXPECT_THAT(error.what(),
                HasSubstr("cannot read configuration 'fail-create'"));
  }
}

TEST(InjectionLoader, ResolvesRelativePath) {
  const TempDir dir;
  const auto path = dir.install(SPECFEMPP_TEST_GOOD_PLUGIN);
  const auto relative = std::filesystem::relative(path);

  const auto plugin =
      specfem::injection::PluginLibrary::open(relative.string());
  EXPECT_EQ(plugin.source(), std::filesystem::canonical(path).string());
}

TEST(InjectionLoader, BareNameIsNotSearched) {
  // Without the static: prefix a bare name is a file in the working
  // directory, never a library search.
  EXPECT_THAT(open_error("libc.so.6"),
              HasSubstr("Cannot resolve injection plugin"));
}

TEST(InjectionLoader, HashIdentifiesBuild) {
  const TempDir dir;
  const auto first = specfem::injection::PluginLibrary::open(
      dir.install(SPECFEMPP_TEST_GOOD_PLUGIN, "first.so"));
  const auto copy = specfem::injection::PluginLibrary::open(
      dir.install(SPECFEMPP_TEST_GOOD_PLUGIN, "copy.so"));
  EXPECT_EQ(first.sha256(), copy.sha256());

  // Any change to the bytes changes the digest.
  const auto changed = dir.install(SPECFEMPP_TEST_GOOD_PLUGIN, "changed.so");
  std::ofstream(changed, std::ios::binary | std::ios::app) << '\0';
  EXPECT_NE(specfem::injection::PluginLibrary::open(changed).sha256(),
            first.sha256());
}

TEST(InjectionLoader, RejectsMissingFile) {
  const TempDir dir;
  EXPECT_THAT(open_error((dir.path() / "missing.so").string()),
              HasSubstr("Cannot resolve injection plugin"));
}

TEST(InjectionLoader, RejectsDirectory) {
  const TempDir dir;
  EXPECT_THAT(open_error(dir.path().string()),
              HasSubstr("is not a regular file"));
}

TEST(InjectionLoader, RejectsWorldWritableFile) {
  const TempDir dir;
  const auto path = dir.install(SPECFEMPP_TEST_GOOD_PLUGIN);
  ::chmod(path.c_str(), 0757);
  const auto error = open_error(path);
  EXPECT_THAT(error, HasSubstr("file '"));
  EXPECT_THAT(error, HasSubstr("is world-writable (mode 757)"));
  EXPECT_THAT(error, HasSubstr("chmod o-w"));
}

TEST(InjectionLoader, RejectsWorldWritableDirectory) {
  const TempDir dir;
  const auto path = dir.install(SPECFEMPP_TEST_GOOD_PLUGIN);
  ::chmod(dir.path().c_str(), 0777);
  const auto error = open_error(path);
  EXPECT_THAT(error, HasSubstr("directory '"));
  EXPECT_THAT(error, HasSubstr("without the sticky bit (mode 777)"));
}

TEST(InjectionLoader, AcceptsGroupSharedPlugin) {
  // A setgid project directory shared by a research group.
  const TempDir dir;
  const auto path = dir.install(SPECFEMPP_TEST_GOOD_PLUGIN);
  ::chmod(path.c_str(), 0775);
  ::chmod(dir.path().c_str(), 02775);
  EXPECT_NO_THROW([[maybe_unused]] const auto plugin =
                      specfem::injection::PluginLibrary::open(path));
}

TEST(InjectionLoader, AcceptsStickyWorldWritableDirectory) {
  // Like /tmp: others can write, but cannot replace this user's file.
  const TempDir dir;
  const auto path = dir.install(SPECFEMPP_TEST_GOOD_PLUGIN);
  ::chmod(dir.path().c_str(), 01777);
  EXPECT_NO_THROW([[maybe_unused]] const auto plugin =
                      specfem::injection::PluginLibrary::open(path));
}

// Message of the std::runtime_error thrown by the privileged checks.
std::string privileged_error(const std::string &path) {
  try {
    specfem::injection_impl::check_permissions(
        std::filesystem::canonical(path).string(), true);
  } catch (const std::runtime_error &error) {
    return error.what();
  }
  return "<no error>";
}

TEST(InjectionLoader, PrivilegedRejectsGroupWritableFile) {
  const TempDir dir;
  const auto path = dir.install(SPECFEMPP_TEST_GOOD_PLUGIN);
  ::chmod(path.c_str(), 0775);
  const auto error = privileged_error(path);
  EXPECT_THAT(error, HasSubstr("file '"));
  EXPECT_THAT(error, HasSubstr("group- or world-writable (mode 775)"));
}

TEST(InjectionLoader, PrivilegedRejectsGroupWritableDirectory) {
  const TempDir dir;
  const auto path = dir.install(SPECFEMPP_TEST_GOOD_PLUGIN);
  ::chmod(dir.path().c_str(), 0770);
  const auto error = privileged_error(path);
  EXPECT_THAT(error, HasSubstr("directory '"));
  EXPECT_THAT(error, HasSubstr("without the sticky bit (mode 770)"));
}

TEST(InjectionLoader, PrivilegedChecksEveryAncestor) {
  // The plugin's own directory is fine; the one above it is not.
  const TempDir dir;
  const auto inner = dir.path() / "inner";
  std::filesystem::create_directory(inner);
  ::chmod(inner.c_str(), 0700);
  const auto path = inner / "plugin.so";
  std::filesystem::copy_file(SPECFEMPP_TEST_GOOD_PLUGIN, path);
  ::chmod(path.c_str(), 0755);
  ::chmod(dir.path().c_str(), 0770);
  const auto error = privileged_error(path.string());
  EXPECT_THAT(error, HasSubstr(std::filesystem::canonical(dir.path()).string() +
                               "' is group- or world-writable"));
}

TEST(InjectionLoader, RejectsMissingManifest) {
  const TempDir dir;
  EXPECT_THAT(open_error(dir.install(SPECFEMPP_TEST_NO_MANIFEST_PLUGIN)),
              HasSubstr("has no plugin manifest"));
}

TEST(InjectionLoader, RejectsWrongManifestAbiVersion) {
  const TempDir dir;
  EXPECT_THAT(open_error(dir.install(SPECFEMPP_TEST_BAD_ABI_PLUGIN)),
              HasSubstr("built for ABI version 99"));
}

TEST(InjectionLoader, RejectsMissingEntryPoint) {
  const TempDir dir;
  EXPECT_THAT(open_error(dir.install(SPECFEMPP_TEST_MISSING_SYMBOL_PLUGIN)),
              HasSubstr("does not export specfempp_injection_plugin()"));
}

TEST(InjectionLoader, RejectsTableManifestMismatch) {
  const TempDir dir;
  const auto error =
      open_error(dir.install(SPECFEMPP_TEST_TABLE_MISMATCH_PLUGIN));
  EXPECT_THAT(error, HasSubstr("invalid function table"));
  EXPECT_THAT(error, HasSubstr("built for ABI version 2"));
}

// In-binary plugin for the static-registration tests; counts live instances.
int static_instances = 0;

void *
static_create(const char *config_path,
              [[maybe_unused]] const specfempp_simulation_info *info,
              [[maybe_unused]] const specfempp_communicator *communicator) {
  if (std::string(config_path) == "fail-create")
    return nullptr;
  ++static_instances;
  return &static_instances;
}
int32_t static_supports([[maybe_unused]] void *self,
                        [[maybe_unused]] specfempp_medium medium,
                        [[maybe_unused]] specfempp_quantity quantity) {
  return 0;
}
specfempp_status
static_setup([[maybe_unused]] void *self,
             [[maybe_unused]] const specfempp_points *points,
             [[maybe_unused]] const specfempp_request *request) {
  return specfempp_status_ok;
}
specfempp_status static_fill([[maybe_unused]] void *self,
                             [[maybe_unused]] int32_t first_step,
                             [[maybe_unused]] const specfempp_buffer *buffer) {
  return specfempp_status_ok;
}
const char *static_last_error(const void *self) {
  return self == nullptr ? "static create failed" : "";
}
void static_destroy([[maybe_unused]] void *self) { --static_instances; }

specfempp_injection_table static_table(const char *name) {
  return specfempp_injection_table{ .abi_version =
                                        SPECFEMPP_INJECTION_ABI_VERSION,
                                    .name = name,
                                    .version = "3.0.0",
                                    .create = static_create,
                                    .supports = static_supports,
                                    .setup = static_setup,
                                    .fill = static_fill,
                                    .last_error = static_last_error,
                                    .destroy = static_destroy };
}

TEST(InjectionLoader, StaticPluginEndToEnd) {
  static const specfempp_injection_table table = static_table("loader-static");
  specfem::injection::register_static_plugin(&table);

  const auto plugin =
      specfem::injection::PluginLibrary::open("static:loader-static");
  EXPECT_EQ(plugin.name(), "loader-static");
  EXPECT_EQ(plugin.version(), "3.0.0");
  EXPECT_EQ(plugin.source(), "static:loader-static");
  {
    const auto instance =
        plugin.create("ok", simulation_info(), communicator());
    EXPECT_EQ(static_instances, 1);
  }
  EXPECT_EQ(static_instances, 0);

  try {
    [[maybe_unused]] const auto instance =
        plugin.create("fail-create", simulation_info(), communicator());
    FAIL() << "create should have thrown";
  } catch (const std::runtime_error &error) {
    EXPECT_THAT(error.what(), HasSubstr("static create failed"));
  }

  // Registering the same name twice is an error.
  static const specfempp_injection_table duplicate =
      static_table("loader-static");
  EXPECT_THROW(specfem::injection::register_static_plugin(&duplicate),
               std::runtime_error);
}

TEST(InjectionLoader, UnknownStaticPluginListsRegistered) {
  static const specfempp_injection_table table = static_table("loader-listed");
  specfem::injection::register_static_plugin(&table);
  const auto error = open_error("static:no-such-plugin");
  EXPECT_THAT(error, HasSubstr("No statically registered injection plugin "
                               "named 'no-such-plugin'"));
  EXPECT_THAT(error, HasSubstr("loader-listed"));
}

TEST(InjectionLoader, RejectsInvalidStaticTable) {
  static specfempp_injection_table table = static_table("loader-invalid");
  table.fill = nullptr;
  try {
    specfem::injection::register_static_plugin(&table);
    FAIL() << "registration should have thrown";
  } catch (const std::runtime_error &error) {
    EXPECT_THAT(error.what(), HasSubstr("required function pointer is null"));
  }
  EXPECT_THROW(specfem::injection::register_static_plugin(nullptr),
               std::runtime_error);
}

} // namespace specfem::injection_loader_test
