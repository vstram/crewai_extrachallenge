from crewai.tools import BaseTool
from typing import List, Type
from pydantic import BaseModel, Field
import re


class MarkdownFormatterInput(BaseModel):
    """Input schema for MarkdownFormatterTool."""
    markdown_content: str = Field(..., description="Markdown content to format and validate")


class MarkdownFormatterTool(BaseTool):
    name: str = "Markdown Formatter Tool"
    description: str = (
        "Formats and validates markdown content to ensure proper structure. "
        "Fixes common issues like missing blank lines after headers, proper spacing, and formatting."
    )
    args_schema: Type[BaseModel] = MarkdownFormatterInput

    def _run(self, markdown_content: str) -> str:
        """Format and validate markdown content."""

        try:
            formatted_content = self._format_markdown(markdown_content)

            # Add clear instruction header for the agent
            instruction_header = (
                "=== FORMATTED MARKDOWN OUTPUT ===\n"
                "INSTRUCTIONS: Copy everything BELOW this line and submit as your final response.\n"
                "DO NOT wrap in backticks or code fences.\n"
                "=================================\n\n"
            )

            # Add validation comment in the actual markdown
            validation_msg = "<!-- FORMATTED BY MARKDOWN FORMATTER TOOL -->\n"

            return f"{instruction_header}{validation_msg}{formatted_content}"

        except Exception as e:
            return f"Error formatting markdown: {str(e)}\n\nOriginal content:\n{markdown_content}"

    def _format_markdown(self, content: str) -> str:
        """Apply markdown formatting rules."""

        # Remove any leading/trailing triple backticks (common agent error)
        content = re.sub(r'^```\n?', '', content, flags=re.MULTILINE)
        content = re.sub(r'\n?```$', '', content, flags=re.MULTILINE)
        content = content.strip()

        # Split into lines for processing
        lines = content.split('\n')
        formatted_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # Handle headers (h1, h2, h3, h4, h5, h6)
            if re.match(r'^#{1,6}\s+', line):
                # Add blank line before header (if not at start and previous line isn't blank)
                if i > 0 and formatted_lines and formatted_lines[-1].strip() != '':
                    formatted_lines.append('')

                # Add the header
                formatted_lines.append(line)

                # Add blank line after header (if next line exists and isn't blank)
                if i + 1 < len(lines) and lines[i + 1].strip() != '':
                    formatted_lines.append('')

            # Handle list items
            elif re.match(r'^[\*\-\+]\s+', line) or re.match(r'^\d+\.\s+', line):
                formatted_lines.append(line)

            # Handle table rows
            elif '|' in line and line.strip().startswith('|'):
                formatted_lines.append(line)

            # Handle code blocks
            elif line.strip().startswith('```'):
                # Add blank line before code block
                if formatted_lines and formatted_lines[-1].strip() != '':
                    formatted_lines.append('')

                formatted_lines.append(line)

                # Find the end of the code block
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    formatted_lines.append(lines[i])
                    i += 1

                # Add the closing ```
                if i < len(lines):
                    formatted_lines.append(lines[i])

                # Add blank line after code block
                if i + 1 < len(lines) and lines[i + 1].strip() != '':
                    formatted_lines.append('')

            # Handle horizontal rules
            elif re.match(r'^[\-\*_]{3,}$', line.strip()):
                # Add blank lines around horizontal rules
                if formatted_lines and formatted_lines[-1].strip() != '':
                    formatted_lines.append('')
                formatted_lines.append(line)
                if i + 1 < len(lines) and lines[i + 1].strip() != '':
                    formatted_lines.append('')

            # Handle blockquotes
            elif line.strip().startswith('>'):
                formatted_lines.append(line)

            # Handle image references
            elif re.match(r'!\[.*\]\(.*\)', line.strip()):
                # Add blank line before image if needed
                if formatted_lines and formatted_lines[-1].strip() != '':
                    formatted_lines.append('')
                # Fix image path to use ./images/ prefix
                fixed_line = self._fix_image_path(line)
                formatted_lines.append(fixed_line)
                # Add blank line after image if needed
                if i + 1 < len(lines) and lines[i + 1].strip() != '':
                    formatted_lines.append('')

            # Handle regular paragraphs and empty lines
            else:
                formatted_lines.append(line)

            i += 1

        # Clean up multiple consecutive blank lines
        final_lines = []
        blank_count = 0

        for line in formatted_lines:
            if line.strip() == '':
                blank_count += 1
                if blank_count <= 2:  # Allow maximum 2 consecutive blank lines
                    final_lines.append(line)
            else:
                blank_count = 0
                final_lines.append(line)

        # Remove trailing blank lines
        while final_lines and final_lines[-1].strip() == '':
            final_lines.pop()

        # Ensure document ends with a single newline (MD047 compliance)
        if final_lines:
            final_lines.append('')
            # Add second empty line to ensure trailing newline is preserved
            final_lines.append('')

        return '\n'.join(final_lines)

    def _fix_image_path(self, line: str) -> str:
        """Fix image paths to use ./images/ prefix."""
        # Pattern to match image markdown: ![alt text](path)
        image_pattern = r'!\[(.*?)\]\((.*?)\)'

        def replace_path(match):
            alt_text = match.group(1)
            path = match.group(2)

            # Extract just the filename if path contains directory separators
            if '/' in path:
                filename = path.split('/')[-1]
            else:
                filename = path

            # Ensure path starts with ./images/
            if not path.startswith('./images/'):
                return f'![{alt_text}](./images/{filename})'
            else:
                return match.group(0)  # Already correct

        return re.sub(image_pattern, replace_path, line)

    def _validate_markdown(self, content: str) -> List[str]:
        """Validate markdown and return list of issues found."""

        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            # Check for headers without blank line after
            if re.match(r'^#{1,6}\s+', line):
                if i < len(lines) and lines[i].strip() != '' and not re.match(r'^#{1,6}\s+', lines[i]):
                    issues.append(f"Line {i}: Header should be followed by a blank line")

            # Check for missing alt text in images
            if re.match(r'!\[\s*\]\(.*\)', line):
                issues.append(f"Line {i}: Image missing alt text")

            # Check for broken image references
            if '![' in line and '](' in line:
                if './images/' not in line:
                    issues.append(f"Line {i}: Image reference should use ./images/ path")

        return issues