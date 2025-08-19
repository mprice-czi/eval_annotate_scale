# Bzlmod Migration Notes

This project has been migrated from the legacy WORKSPACE system to Bazel's modern Bzlmod (MODULE.bazel) system.

## Key Changes

### Files Added/Modified
- `MODULE.bazel` - New dependency management file (replaces WORKSPACE)
- `.bazelversion` - Pins Bazel version to 8.3.0
- `.bazelrc` - Configuration with Bzlmod enabled
- `WORKSPACE.legacy` - Backup of old WORKSPACE file

### Benefits of Bzlmod
- **Hermetic Dependencies**: Better isolation and reproducibility
- **Version Resolution**: Automatic dependency version conflict resolution
- **Module Registry**: Central registry for Bazel modules
- **Simpler Configuration**: More declarative dependency management
- **Future-Proof**: Modern Bazel approach going forward

### Migration Commands
```bash
# Build with new system (automatic with .bazelrc)
bazel build //...

# Query dependencies
bazel mod deps

# Show module graph  
bazel mod graph

# Explain dependency resolution
bazel mod explain <module_name>
```

### Troubleshooting
- If builds fail, ensure `.bazelrc` enables Bzlmod with `--enable_bzlmod`
- Use `bazel mod` commands to debug dependency issues
- Check MODULE.bazel syntax for proper module declarations

### Legacy Support
The old WORKSPACE file has been preserved as `WORKSPACE.legacy` for reference. It can be restored if needed for compatibility, but Bzlmod is recommended for all new development.