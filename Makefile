# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -O2 -Iinclude

# Directories
SRCDIR := src
INCDIR := include
BUILDDIR := build
BINDIR := bin

# Files
SOURCES := $(wildcard $(SRCDIR)/*.cc)
OBJECTS := $(patsubst $(SRCDIR)/%.cc, $(BUILDDIR)/%.o, $(SOURCES))
EXECUTABLE := $(BINDIR)/finn

# Default target
all: $(EXECUTABLE)

# Linking the executable
$(EXECUTABLE): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Compiling source files to object files
$(BUILDDIR)/%.o: $(SRCDIR)/%.cc | $(BUILDDIR)
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create build directory if it doesn't exist
$(BUILDDIR):
	@mkdir -p $(BUILDDIR)

# Clean up build artifacts
clean:
	@echo "Cleaning up..."
	@rm -rf $(BUILDDIR) $(BINDIR)

# Phony targets
.PHONY: all clean
