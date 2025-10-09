import csv

# ============ COLOR CONFIGURATION ============
# Base Colors
BG_MAIN = "#fcfafa"
BG_CARD = "#fcfafa"
BG_TAG = "#E8E8E4"
BORDER_LIGHT = "#F5F5F0"
BORDER_MEDIUM = "#F5F5F0"
TEXT_MAIN = "#414545"
TEXT_TITLE = "#414545"
TEXT_FOOTER = "#414545"

# Tag Text Colors
TAG_SINGLE_TASK = "#1565C0"
TAG_SINGLE_AGENT = "#6A1B9A"
TAG_MULTI_TASK = "#2E7D32"
TAG_MULTI_AGENT = "#E65100"
TAG_3D_ROBOTICS = "#C2185B"
TAG_LOGIC = "#00695C"
TAG_PACKING = "#F57F17"
TAG_ROUTING = "#01579B"
TAG_PLANNING = "#558B2F"
TAG_SWARMS = "#BF360C"
TAG_OPEN_ENDED = "#4527A0"

# Tag Background Colors
TAG_BG_SINGLE_TASK = "#E3F2FD"
TAG_BG_SINGLE_AGENT = "#F3E5F5"
TAG_BG_MULTI_TASK = "#E8F5E9"
TAG_BG_MULTI_AGENT = "#FFF3E0"
TAG_BG_3D_ROBOTICS = "#FCE4EC"
TAG_BG_LOGIC = "#E0F2F1"
TAG_BG_PACKING = "#FFF9C4"
TAG_BG_ROUTING = "#E1F5FE"
TAG_BG_PLANNING = "#F1F8E9"
TAG_BG_SWARMS = "#FBE9E7"
TAG_BG_OPEN_ENDED = "#EDE7F6"

# Tag Border Colors
TAG_BORDER_SINGLE_TASK = "#BBDEFB"
TAG_BORDER_SINGLE_AGENT = "#E1BEE7"
TAG_BORDER_MULTI_TASK = "#C8E6C9"
TAG_BORDER_MULTI_AGENT = "#FFE0B2"
TAG_BORDER_3D_ROBOTICS = "#F8BBD0"
TAG_BORDER_LOGIC = "#B2DFDB"
TAG_BORDER_PACKING = "#FFF59D"
TAG_BORDER_ROUTING = "#B3E5FC"
TAG_BORDER_PLANNING = "#DCEDC8"
TAG_BORDER_SWARMS = "#FFCCBC"
TAG_BORDER_OPEN_ENDED = "#D1C4E9"
# ============================================

def convert_to_raw_url(url):
    """Convert GitHub blob URLs to raw URLs"""
    if 'github.com' in url and '/blob/' in url:
        # Convert github.com/user/repo/blob/branch/path to raw.githubusercontent.com/user/repo/branch/path
        url = url.replace('github.com', 'raw.githubusercontent.com')
        url = url.replace('/blob/', '/')
    return url

def format_environment_name(name):
    """Format environment name by splitting on underscore and capitalizing"""
    if not name:
        return name
    # Split by underscore and capitalize each word
    words = name.split('_')
    return ' '.join(word.capitalize() for word in words)

def get_tag_class(tag):
    """Convert tag name to CSS class name"""
    # Convert to lowercase and replace spaces with hyphens
    class_name = tag.lower().replace(' ', '-')
    return f"tag tag-{class_name}"

def generate_html(csv_file, output_file):
    # Read CSV and collect all environments with tags
    environments = []
    all_tags = set()

    with open(csv_file, 'r') as f:
        # Read raw CSV to get all values including those in unnamed columns
        lines = f.readlines()
        if not lines:
            return

        # Parse header
        header = lines[0].strip().split(',')
        env_url_index = header.index('environment_url') if 'environment_url' in header else -1

        if env_url_index == -1:
            return

        # Tag columns start after environment_url
        tag_start_index = env_url_index + 1

        # Find column indices
        domain_index = header.index('domain') if 'domain' in header else 0
        domain_url_index = header.index('domain_url') if 'domain_url' in header else 1
        env_index = header.index('environment') if 'environment' in header else 2

        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue

            domain = parts[domain_index] if len(parts) > domain_index else ''
            domain_url = parts[domain_url_index] if len(parts) > domain_url_index else ''
            environment = parts[env_index] if len(parts) > env_index else ''
            environment_url = parts[env_url_index] if len(parts) > env_url_index else ''

            # Collect tags from columns after environment_url
            tags = []
            for i in range(tag_start_index, len(parts)):
                tag_value = parts[i].strip()
                if tag_value:
                    tags.append(tag_value)
                    all_tags.add(tag_value)

            environments.append({
                'domain': domain,
                'domain_url': domain_url,
                'environment': environment,
                'environment_url': convert_to_raw_url(environment_url),
                'tags': tags
            })

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NiceWebRL - Environments</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: {TEXT_MAIN};
            background: {BG_MAIN};
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 32px 16px;
        }}

        header {{
            text-align: center;
            margin-bottom: 32px;
            padding: 48px 16px 0 16px;
            background: transparent;
        }}

        h1 {{
            font-size: 3.5em;
            margin-bottom: 10px;
            color: {TEXT_TITLE};
            font-weight: 700;
        }}

        .subtitle {{
            font-size: 1.3em;
            color: {TEXT_TITLE};
            font-weight: 400;
        }}

        .environments-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 32px;
            background: transparent;
            padding: 0;
        }}

        .environment-card {{
            background: transparent;
            border-radius: 0;
            overflow: visible;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border: none;
            display: flex;
            flex-direction: column;
            text-decoration: none;
            color: inherit;
            box-shadow: none;
        }}

        .environment-card:hover {{
            transform: translateY(-4px);
            cursor: pointer;
        }}

        .domain-badge {{
            padding: 0 0 8px 0;
            background: transparent;
            color: {TEXT_TITLE};
            font-size: 1.1em;
            font-weight: 700;
            text-align: center;
            flex-shrink: 0;
            letter-spacing: 0.3px;
        }}

        .image-container {{
            height: 250px;
            background: transparent;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0;
            flex-shrink: 0;
        }}

        .environment-image {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}

        .environment-name {{
            padding: 0 0 12px 0;
            font-weight: 500;
            font-size: 0.85em;
            color: {TEXT_TITLE};
            text-align: center;
            background: transparent;
        }}

        .tag-filter-section {{
            background: transparent;
            padding: 0 0 32px 0;
            margin-bottom: 32px;
        }}

        .tag-filter-section h3 {{
            margin-bottom: 16px;
            color: {TEXT_TITLE};
            font-size: 1.3em;
            font-weight: 600;
        }}

        .tag-filters {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}

        .tag-filter {{
            padding: 10px 20px;
            border-radius: 24px;
            font-size: 0.9em;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            user-select: none;
            font-weight: 500;
            opacity: 0.6;
            border: none;
        }}

        .tag-filter:hover {{
            opacity: 0.8;
            transform: translateY(-2px);
        }}

        .tag-filter.active {{
            opacity: 1;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        }}

        .environment-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 12px 0;
            background: transparent;
            min-height: 40px;
            align-items: center;
            flex-shrink: 0;
            justify-content: center;
        }}

        .tag {{
            padding: 6px 12px;
            background: {BG_TAG};
            color: {TEXT_TITLE};
            border-radius: 16px;
            font-size: 0.75em;
            font-weight: 600;
            border: none;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .tag:hover {{
            opacity: 0.8;
            transform: scale(1.05);
        }}

        .tag-single-task {{
            background: {TAG_BG_SINGLE_TASK};
            color: {TAG_SINGLE_TASK};
        }}

        .tag-single-agent {{
            background: {TAG_BG_SINGLE_AGENT};
            color: {TAG_SINGLE_AGENT};
        }}

        .tag-multi-task {{
            background: {TAG_BG_MULTI_TASK};
            color: {TAG_MULTI_TASK};
        }}

        .tag-multi-agent {{
            background: {TAG_BG_MULTI_AGENT};
            color: {TAG_MULTI_AGENT};
        }}

        .tag-3d-robotics {{
            background: {TAG_BG_3D_ROBOTICS};
            color: {TAG_3D_ROBOTICS};
        }}

        .tag-logic {{
            background: {TAG_BG_LOGIC};
            color: {TAG_LOGIC};
        }}

        .tag-packing {{
            background: {TAG_BG_PACKING};
            color: {TAG_PACKING};
        }}

        .tag-routing {{
            background: {TAG_BG_ROUTING};
            color: {TAG_ROUTING};
        }}

        .tag-planning {{
            background: {TAG_BG_PLANNING};
            color: {TAG_PLANNING};
        }}

        .tag-swarms {{
            background: {TAG_BG_SWARMS};
            color: {TAG_SWARMS};
        }}

        .tag-open-ended {{
            background: {TAG_BG_OPEN_ENDED};
            color: {TAG_OPEN_ENDED};
        }}

        .tag-filter.tag-single-task {{
            background: {BG_TAG};
            color: {TAG_SINGLE_TASK};
        }}

        .tag-filter.tag-single-agent {{
            background: {BG_TAG};
            color: {TAG_SINGLE_AGENT};
        }}

        .tag-filter.tag-multi-task {{
            background: {BG_TAG};
            color: {TAG_MULTI_TASK};
        }}

        .tag-filter.tag-multi-agent {{
            background: {BG_TAG};
            color: {TAG_MULTI_AGENT};
        }}

        .tag-filter.tag-3d-robotics {{
            background: {BG_TAG};
            color: {TAG_3D_ROBOTICS};
        }}

        .tag-filter.tag-logic {{
            background: {BG_TAG};
            color: {TAG_LOGIC};
        }}

        .tag-filter.tag-packing {{
            background: {BG_TAG};
            color: {TAG_PACKING};
        }}

        .tag-filter.tag-routing {{
            background: {BG_TAG};
            color: {TAG_ROUTING};
        }}

        .tag-filter.tag-planning {{
            background: {BG_TAG};
            color: {TAG_PLANNING};
        }}

        .tag-filter.tag-swarms {{
            background: {BG_TAG};
            color: {TAG_SWARMS};
        }}

        .tag-filter.tag-open-ended {{
            background: {BG_TAG};
            color: {TAG_OPEN_ENDED};
        }}

        .environment-card.hidden {{
            display: none;
        }}

        footer {{
            text-align: center;
            margin-top: 48px;
            padding: 16px;
            color: {TEXT_FOOTER};
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>NiceWebRL</h1>
            <p class="subtitle">A Python library for quickly making human subject experiments with machine learning models and environments.</p>
        </header>

        <div class="tag-filter-section">
            <h3>Filter by Tags</h3>
            <div class="tag-filters">
"""

    # Add tag filters
    sorted_tags = sorted(all_tags)
    for tag in sorted_tags:
        tag_class = get_tag_class(tag).replace('tag ', '')  # Get just the specific tag class
        html += f"""                <div class="tag-filter {tag_class}" onclick="toggleTag('{tag}')">{tag}</div>
"""

    html += """            </div>
        </div>

        <div class="environments-grid">
"""

    # Add each environment
    for env in environments:
        domain = env['domain']
        domain_url = env['domain_url']
        env_name = env['environment']
        env_url = env['environment_url']
        formatted_name = format_environment_name(env_name)
        tags = env['tags']
        tags_str = ','.join(tags)

        if domain_url:
            html += f"""            <a href="{domain_url}" target="_blank" class="environment-card" data-tags="{tags_str}">
"""
        else:
            html += f"""            <div class="environment-card" data-tags="{tags_str}">
"""

        html += f"""                <div class="domain-badge">{domain}</div>
"""
        if formatted_name:
            html += f"""                <div class="environment-name">{formatted_name}</div>
"""

        html += f"""                <div class="image-container">
                    <img src="{env_url}" alt="{env_name}" class="environment-image" onerror="this.parentElement.style.display='none'">
                </div>
"""

        html += """                <div class="environment-tags">
"""
        for tag in tags:
            tag_class = get_tag_class(tag)
            html += f"""                    <span class="{tag_class}" onclick="event.preventDefault(); event.stopPropagation(); toggleTagFromCard('{tag}');">{tag}</span>
"""
        html += """                </div>
"""

        if domain_url:
            html += """            </a>
"""
        else:
            html += """            </div>
"""

    html += """        </div>
"""

    # Close HTML
    html += """
    </div>

    <script>
        let activeTags = new Set();

        function toggleTag(tag) {
            const filterElement = event.target;

            if (activeTags.has(tag)) {
                activeTags.delete(tag);
                filterElement.classList.remove('active');
            } else {
                activeTags.add(tag);
                filterElement.classList.add('active');
            }

            filterEnvironments();
        }

        function toggleTagFromCard(tag) {
            if (activeTags.has(tag)) {
                activeTags.delete(tag);
            } else {
                activeTags.add(tag);
            }

            // Update filter button state
            const filterButtons = document.querySelectorAll('.tag-filter');
            filterButtons.forEach(btn => {
                if (btn.textContent === tag) {
                    if (activeTags.has(tag)) {
                        btn.classList.add('active');
                    } else {
                        btn.classList.remove('active');
                    }
                }
            });

            filterEnvironments();
        }

        function filterEnvironments() {
            const cards = document.querySelectorAll('.environment-card');

            if (activeTags.size === 0) {
                // Show all if no filters active
                cards.forEach(card => card.classList.remove('hidden'));
                return;
            }

            cards.forEach(card => {
                const cardTags = card.dataset.tags.split(',').filter(t => t.length > 0);
                const hasAllTags = Array.from(activeTags).every(tag => cardTags.includes(tag));

                if (hasAllTags) {
                    card.classList.remove('hidden');
                } else {
                    card.classList.add('hidden');
                }
            });
        }
    </script>
</body>
</html>
"""

    # Write to file
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"Generated {output_file}")

if __name__ == '__main__':
    generate_html('environments.csv', 'index.html')
