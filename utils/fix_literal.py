import re

with open("C:/Users/lenovo/hwin_net/utils/config.py", "r") as f:
    content = f.read()

# Replace all Literal[...] with str for OmegaConf compatibility
# We will keep the valid values in comments/description instead

# Find and replace Literal patterns
content = re.sub(
    r"Literal\[([^\]]+)\]",
    lambda m: "str  # " + m.group(1).replace("\"", ""),
    content
)

# Also fix the root config fields
content = re.sub(
    r"log_level: str  # debug, info, warning, error",
    "log_level: str = field(default=\"info\", metadata={\"theorem_ref\": \"impl\", \"description\": \"Logging level\"})",
    content
)

with open("C:/Users/lenovo/hwin_net/utils/config.py", "w") as f:
    f.write(content)

print("Replaced Literal types with str")
