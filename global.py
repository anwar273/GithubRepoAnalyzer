import streamlit as st
import os
import tempfile
import shutil
import subprocess
import json
import uuid
import pandas as pd
import matplotlib.pyplot as plt
import time
import re
import base64
from datetime import datetime
from git import Repo
from fpdf import FPDF
import requests
from langchain_community.llms import OpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import markdown
import yaml
from pathlib import Path
import zipfile
import logging
from mistral_llm import MistralLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="GitHub Repository Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .vulnerability-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .vulnerability-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .vulnerability-low {
        color: #4b96ff;
        font-weight: bold;
    }
    .report-section {
        background-color: white;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
        border: 1px solid #ddd;
    }
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'repo_path' not in st.session_state:
    st.session_state.repo_path = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'repo_analysis' not in st.session_state:
    st.session_state.repo_analysis = None
if 'vulnerabilities' not in st.session_state:
    st.session_state.vulnerabilities = None
if 'repo_summary' not in st.session_state:
    st.session_state.repo_summary = None
if 'generated_pipeline' not in st.session_state:
    st.session_state.generated_pipeline = None
if 'report_data' not in st.session_state:
    st.session_state.report_data = {}
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = None
if 'llm_client' not in st.session_state:
    st.session_state.llm_client = None


# Title and description
st.title("🔍 GitHub Repository Analyzer")
st.markdown("Analyze GitHub repositories for vulnerabilities, understand code context, and generate deployment pipelines")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Setup", "Analysis", "Results", "Deployment"])

class LLMProvider:
    """Base class for LLM providers"""
    def __init__(self):
        self.client = None
    
    def get_client(self):
        return self.client
    
    def analyze_text(self, prompt, max_tokens=1000):
        pass


class OllamaProvider(LLMProvider):
    """Ollama local model provider"""
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api"
        
        # Check if Ollama is running
        try:
            response = requests.get(f"{self.base_url}/tags")
            if response.status_code == 200:
                available_models = [model['name'] for model in response.json()['models']]
                if model_name not in available_models:
                    st.warning(f"Model {model_name} not found in Ollama. Please run 'ollama pull {model_name}' first.")
                self.client = model_name
            else:
                st.error("Failed to connect to Ollama API")
        except requests.exceptions.ConnectionError:
            st.error("Ollama service is not running. Please start Ollama first.")
    
    def analyze_text(self, prompt, max_tokens=1000):
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "max_tokens": max_tokens
                }
            )
            if response.status_code == 200:
                return response.json()['response']
            else:
                st.error(f"Error from Ollama API: {response.text}")
                return None
        except Exception as e:
            st.error(f"Error calling Ollama API: {str(e)}")
            return None


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    def __init__(self, api_key, model="gpt-4"):
        super().__init__()
        self.api_key = api_key
        self.model = model
        
        try:
            import openai
            openai.api_key = api_key
            self.client = openai
        except ImportError:
            st.error("OpenAI package not installed. Please install it with 'pip install openai'")
    
    def analyze_text(self, prompt, max_tokens=1000):
        try:
            response = self.client.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error calling OpenAI API: {str(e)}")
            return None


class AnthropicProvider(LLMProvider):
    """Anthropic API provider"""
    def __init__(self, api_key, model="claude-3-opus-20240229"):
        super().__init__()
        self.api_key = api_key
        self.model = model
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            st.error("Anthropic package not installed. Please install it with 'pip install anthropic'")
    
    def analyze_text(self, prompt, max_tokens=1000):
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            st.error(f"Error calling Anthropic API: {str(e)}")
            return None


class RepositoryHandler:
    """Handle GitHub repository operations"""
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {self.temp_dir}")
    
    def clone_repository(self, repo_url, auth_token=None):
        """Clone a repository to the temporary directory"""
        try:
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            repo_path = os.path.join(self.temp_dir, repo_name)
            
            if auth_token:
                # For private repositories
                auth_repo_url = repo_url.replace("https://", f"https://{auth_token}@")
                Repo.clone_from(auth_repo_url, repo_path)
            else:
                # For public repositories
                Repo.clone_from(repo_url, repo_path)
            
            logger.info(f"Repository cloned to {repo_path}")
            return repo_path
        except Exception as e:
            logger.error(f"Error cloning repository: {str(e)}")
            st.error(f"Error cloning repository: {str(e)}")
            return None
    
    def cleanup(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)
        logger.info(f"Removed temporary directory: {self.temp_dir}")


class VulnerabilityScanner:
    """Scan repository for vulnerabilities"""
    def __init__(self, repo_path, llm_provider):
        self.repo_path = repo_path
        self.llm_provider = llm_provider
    
    def get_file_content(self, file_path):
        """Get the content of a file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return ""
    
    def get_file_list(self):
        """Get a list of all files in the repository"""
        file_list = []
        for root, dirs, files in os.walk(self.repo_path):
            # Skip .git directory and other hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if not file.startswith('.'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, self.repo_path)
                    file_list.append(relative_path)
        
        return file_list
    
    def detect_language(self, file_path):
        """Detect the programming language of a file based on its extension"""
        extensions = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.c': 'C',
            '.cpp': 'C++',
            '.cs': 'C#',
            '.go': 'Go',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.rs': 'Rust',
            '.sh': 'Shell',
            '.html': 'HTML',
            '.css': 'CSS',
            '.sql': 'SQL',
            '.md': 'Markdown',
            '.json': 'JSON',
            '.yml': 'YAML',
            '.yaml': 'YAML',
            '.xml': 'XML',
            '.toml': 'TOML',
            '.Dockerfile': 'Dockerfile',
        }
        
        _, ext = os.path.splitext(file_path)
        if file_path.endswith('Dockerfile'):
            return 'Dockerfile'
        return extensions.get(ext.lower(), 'Unknown')
    
    def analyze_file_for_vulnerabilities(self, file_path, language):
        """Analyze a file for vulnerabilities using LLM"""
        full_path = os.path.join(self.repo_path, file_path)
        content = self.get_file_content(full_path)
        
        # Skip empty files or files that are too large
        if not content or len(content) > 100000:  # Skip files larger than 100KB
            return []
        
        # Create a prompt for the LLM
        prompt = f"""
        Analyze the following {language} code for security vulnerabilities, bad practices, and code smells.
        Focus on the following vulnerability types:
        1. SQL injection
        2. Cross-site scripting (XSS)
        3. Hard-coded secrets or credentials
        4. Insecure cryptographic practices
        5. Path traversal
        6. Command injection
        7. Insecure deserialization
        8. Improper input validation
        9. Insecure direct object references
        10. Improper error handling

        For each vulnerability found, provide:
        - A brief description of the vulnerability
        - The severity level (High, Medium, Low)
        - The line number or range where the vulnerability is located
        - A recommendation for fixing the issue

        Format your response as a JSON array of objects, each with these fields:
        - vulnerability_type: string
        - description: string
        - severity: "High", "Medium", or "Low"
        - line_numbers: array of numbers or ranges (e.g. [10] or [15-20])
        - recommendation: string

        If no vulnerabilities are found, return an empty array.

        CODE TO ANALYZE:
        ```{language}
        {content[:50000]}  # Limit to first 50K characters
        ```

        JSON RESPONSE:
        """
        
        try:
            response = self.llm_provider.analyze_text(prompt, max_tokens=2000)
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                vulnerabilities = json.loads(json_str)
                
                # Add file path to each vulnerability
                for vuln in vulnerabilities:
                    vuln['file_path'] = file_path
                
                return vulnerabilities
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            return []
    
    def scan_repository(self, max_files=50):
        """Scan the repository for vulnerabilities"""
        try:
            file_list = self.get_file_list()
            
            # Limit the number of files to scan
            if len(file_list) > max_files:
                st.warning(f"Repository contains {len(file_list)} files. Limiting scan to {max_files} files.")
                file_list = file_list[:max_files]
            
            st.info(f"Scanning {len(file_list)} files for vulnerabilities...")
            
            all_vulnerabilities = []
            progress_bar = st.progress(0)
            
            for i, file_path in enumerate(file_list):
                language = self.detect_language(file_path)
                if language != 'Unknown':
                    vulnerabilities = self.analyze_file_for_vulnerabilities(file_path, language)
                    all_vulnerabilities.extend(vulnerabilities)
                
                # Update progress bar
                progress_bar.progress((i + 1) / len(file_list))
            
            return all_vulnerabilities
            
        except Exception as e:
            logger.error(f"Error scanning repository: {str(e)}")
            st.error(f"Error scanning repository: {str(e)}")
            return []


class CodeContextAnalyzer:
    """Analyze repository context and structure"""
    def __init__(self, repo_path, llm_provider):
        self.repo_path = repo_path
        self.llm_provider = llm_provider
    
    def get_repository_structure(self):
        """Get the structure of the repository"""
        structure = []
        for root, dirs, files in os.walk(self.repo_path):
            # Skip .git directory and other hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            level = root.replace(self.repo_path, '').count(os.sep)
            indent = ' ' * 4 * level
            structure.append(f"{indent}{os.path.basename(root)}/")
            
            sub_indent = ' ' * 4 * (level + 1)
            for file in files:
                if not file.startswith('.'):
                    structure.append(f"{sub_indent}{file}")
        
        return '\n'.join(structure)
    
    def get_key_files(self):
        """Identify key files in the repository"""
        key_files = []
        key_file_patterns = [
            'README.md',
            'setup.py',
            'requirements.txt',
            'package.json',
            'Dockerfile',
            'docker-compose.yml',
            '.github/workflows/*.yml',
            '.gitlab-ci.yml',
            'Jenkinsfile',
            'main.py',
            'app.py',
            'index.js',
            'app.js',
            'main.go',
            'pom.xml',
            'build.gradle',
        ]
        
        for pattern in key_file_patterns:
            if '*' in pattern:
                dir_path = os.path.join(self.repo_path, os.path.dirname(pattern))
                file_pattern = os.path.basename(pattern)
                if os.path.exists(dir_path):
                    for file in os.listdir(dir_path):
                        if file.endswith(file_pattern.replace('*', '')):
                            key_files.append(os.path.join(os.path.dirname(pattern), file))
            else:
                file_path = os.path.join(self.repo_path, pattern)
                if os.path.exists(file_path):
                    key_files.append(pattern)
        
        return key_files
    
    def analyze_repository_context(self):
        """Analyze the repository context and structure"""
        try:
            structure = self.get_repository_structure()
            key_files = self.get_key_files()
            
            key_file_contents = {}
            for file in key_files:
                try:
                    with open(os.path.join(self.repo_path, file), 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Limit content to prevent token limits
                        key_file_contents[file] = content[:5000] if len(content) > 5000 else content
                except Exception as e:
                    logger.error(f"Error reading file {file}: {str(e)}")
            
            # Create a prompt for the LLM
            prompt = f"""
            Analyze the following GitHub repository structure and key files to understand the project's context.
            
            REPOSITORY STRUCTURE:
            ```
            {structure[:5000]}  # Limit to first 5000 characters
            ```
            
            KEY FILES:
            """
            
            for file, content in key_file_contents.items():
                prompt += f"""
                {file}:
                ```
                {content}
                ```
                """
            
            prompt += """
            Based on the repository structure and key files, provide:
            1. A summary of what the project does
            2. The primary purpose of the codebase
            3. Key components and their functions
            4. Technologies and frameworks used
            5. The project's architecture (if identifiable)
            
            Format your response as a JSON object with these fields:
            - project_name: string
            - project_summary: string
            - primary_purpose: string
            - key_components: array of objects, each with 'name' and 'function' fields
            - technologies: array of strings
            - architecture: string
            
            JSON RESPONSE:
            """
            
            response = self.llm_provider.analyze_text(prompt, max_tokens=2000)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                analysis = json.loads(json_str)
                return analysis
            else:
                return {
                    "project_name": "Unknown",
                    "project_summary": "Could not analyze repository context",
                    "primary_purpose": "Unknown",
                    "key_components": [],
                    "technologies": [],
                    "architecture": "Unknown"
                }
                
        except Exception as e:
            logger.error(f"Error analyzing repository context: {str(e)}")
            st.error(f"Error analyzing repository context: {str(e)}")
            return {
                "project_name": "Unknown",
                "project_summary": f"Error analyzing repository: {str(e)}",
                "primary_purpose": "Unknown",
                "key_components": [],
                "technologies": [],
                "architecture": "Unknown"
            }


class DeploymentPipelineGenerator:
    """Generate deployment pipelines for different CI/CD platforms"""
    def __init__(self, repo_analysis, llm_provider):
        self.repo_analysis = repo_analysis
        self.llm_provider = llm_provider
    
    def generate_github_actions_workflow(self):
        """Generate a GitHub Actions workflow file"""
        prompt = f"""
        Generate a GitHub Actions workflow file for continuous integration and deployment of the following project:
        
        Project Summary: {self.repo_analysis['project_summary']}
        Primary Purpose: {self.repo_analysis['primary_purpose']}
        Technologies: {', '.join(self.repo_analysis['technologies'])}
        
        The workflow should:
        1. Run on push to main branch and pull requests
        2. Set up the appropriate environment based on the technologies used
        3. Install dependencies
        4. Run tests
        5. Build a Docker container
        6. Deploy to AWS, Azure, or Google Cloud (choose the most appropriate based on the project)
        
        Include appropriate environment variables and secrets management.
        
        Provide the complete workflow file in YAML format.
        """
        
        response = self.llm_provider.analyze_text(prompt, max_tokens=2000)
        
        # Extract YAML from response
        yaml_match = re.search(r'```yaml\n(.*?)```', response, re.DOTALL)
        if yaml_match:
            return yaml_match.group(1)
        else:
            yaml_match = re.search(r'```yml\n(.*?)```', response, re.DOTALL)
            if yaml_match:
                return yaml_match.group(1)
            else:
                return response
    
    def generate_gitlab_ci_pipeline(self):
        """Generate a GitLab CI pipeline configuration"""
        prompt = f"""
        Generate a GitLab CI pipeline configuration for continuous integration and deployment of the following project:
        
        Project Summary: {self.repo_analysis['project_summary']}
        Primary Purpose: {self.repo_analysis['primary_purpose']}
        Technologies: {', '.join(self.repo_analysis['technologies'])}
        
        The pipeline should:
        1. Run on push to main branch and merge requests
        2. Set up the appropriate environment based on the technologies used
        3. Install dependencies
        4. Run tests
        5. Build a Docker container
        6. Deploy to AWS, Azure, or Google Cloud (choose the most appropriate based on the project)
        
        Include appropriate environment variables and secrets management.
        
        Provide the complete .gitlab-ci.yml file.
        """
        
        response = self.llm_provider.analyze_text(prompt, max_tokens=2000)
        
        # Extract YAML from response
        yaml_match = re.search(r'```yaml\n(.*?)```', response, re.DOTALL)
        if yaml_match:
            return yaml_match.group(1)
        else:
            yaml_match = re.search(r'```yml\n(.*?)```', response, re.DOTALL)
            if yaml_match:
                return yaml_match.group(1)
            else:
                return response
    
    def generate_jenkins_pipeline(self):
        """Generate a Jenkins pipeline configuration"""
        prompt = f"""
        Generate a Jenkins pipeline (Jenkinsfile) for continuous integration and deployment of the following project:
        
        Project Summary: {self.repo_analysis['project_summary']}
        Primary Purpose: {self.repo_analysis['primary_purpose']}
        Technologies: {', '.join(self.repo_analysis['technologies'])}
        
        The pipeline should:
        1. Run on push to main branch
        2. Set up the appropriate environment based on the technologies used
        3. Install dependencies
        4. Run tests
        5. Build a Docker container
        6. Deploy to AWS, Azure, or Google Cloud (choose the most appropriate based on the project)
        
        Include appropriate environment variables and secrets management.
        
        Provide the complete Jenkinsfile in declarative pipeline syntax.
        """
        
        response = self.llm_provider.analyze_text(prompt, max_tokens=2000)
        
        # Extract groovy from response
        groovy_match = re.search(r'```groovy\n(.*?)```', response, re.DOTALL)
        if groovy_match:
            return groovy_match.group(1)
        else:
            # Try without language specification
            groovy_match = re.search(r'```\n(.*?)```', response, re.DOTALL)
            if groovy_match:
                return groovy_match.group(1)
            else:
                return response
    
    def generate_dockerfile(self):
        """Generate a Dockerfile for the project"""
        prompt = f"""
        Generate a Dockerfile for the following project:
        
        Project Summary: {self.repo_analysis['project_summary']}
        Primary Purpose: {self.repo_analysis['primary_purpose']}
        Technologies: {', '.join(self.repo_analysis['technologies'])}
        
        The Dockerfile should:
        1. Use an appropriate base image
        2. Install dependencies
        3. Copy the project files
        4. Set up the environment
        5. Expose necessary ports
        6. Define the entry point
        
        Provide the complete Dockerfile with detailed comments explaining each step.
        """
        
        response = self.llm_provider.analyze_text(prompt, max_tokens=2000)
        
        # Extract dockerfile from response
        dockerfile_match = re.search(r'```dockerfile\n(.*?)```', response, re.DOTALL)
        if dockerfile_match:
            return dockerfile_match.group(1)
        else:
            # Try without language specification
            dockerfile_match = re.search(r'```\n(.*?)```', response, re.DOTALL)
            if dockerfile_match:
                return dockerfile_match.group(1)
            else:
                return response
    
    def generate_all_pipelines(self):
        """Generate all deployment pipelines"""
        return {
            "github_actions": self.generate_github_actions_workflow(),
            "gitlab_ci": self.generate_gitlab_ci_pipeline(),
            "jenkins": self.generate_jenkins_pipeline(),
            "dockerfile": self.generate_dockerfile()
        }


class ReportGenerator:
    """Generate reports in various formats"""
    def __init__(self, repo_analysis, vulnerabilities, pipelines):
        self.repo_analysis = repo_analysis
        self.vulnerabilities = vulnerabilities
        self.pipelines = pipelines
    
    def generate_markdown_report(self):
        """Generate a markdown report"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Count vulnerabilities by severity
        severity_counts = {"High": 0, "Medium": 0, "Low": 0}
        for vuln in self.vulnerabilities:
            severity_counts[vuln['severity']] += 1
        
        md = f"""# GitHub Repository Analysis Report
        
Generated on: {now}

## Project Overview

- **Project Name**: {self.repo_analysis['project_name']}
- **Project Summary**: {self.repo_analysis['project_summary']}
- **Primary Purpose**: {self.repo_analysis['primary_purpose']}
- **Architecture**: {self.repo_analysis['architecture']}

### Key Components

| Component | Function |
|-----------|----------|
"""
        
        for component in self.repo_analysis['key_components']:
            md += f"| {component['name']} | {component['function']} |\n"
        
        md += """
### Technologies Used

"""
        
        for tech in self.repo_analysis['technologies']:
            md += f"- {tech}\n"
        
        md += f"""
## Vulnerability Analysis

Total vulnerabilities found: {len(self.vulnerabilities)}
- High severity: {severity_counts['High']}
- Medium severity: {severity_counts['Medium']}
- Low severity: {severity_counts['Low']}

### Detailed Vulnerabilities

"""
        
        if len(self.vulnerabilities) > 0:
            for vuln in self.vulnerabilities:
                line_numbers = ', '.join([str(ln) for ln in vuln['line_numbers']]) if isinstance(vuln['line_numbers'], list) else vuln['line_numbers']
                md += f"""
#### {vuln['vulnerability_type']} ({vuln['severity']})

- **File**: {vuln['file_path']}
- **Line(s)**: {line_numbers}
- **Description**: {vuln['description']}
- **Recommendation**: {vuln['recommendation']}

"""
        else:
            md += "No vulnerabilities found.\n"
        
        md += """
## Deployment Pipelines

### GitHub Actions Workflow

```yaml
"""
        
        md += self.pipelines['github_actions']
        
        md += """
```

### GitLab CI Pipeline

```yaml
"""
        
        md += self.pipelines['gitlab_ci']
        
        md += """
```

### Jenkins Pipeline

```groovy
"""
        
        md += self.pipelines['jenkins']
        
        md += """
```

### Dockerfile

```dockerfile
"""
        
        md += self.pipelines['dockerfile']
        
        md += """
```

## Recommendations

1. Address high severity vulnerabilities first
2. Implement the suggested fixes for each vulnerability
3. Set up the CI/CD pipeline to automate testing and deployment
4. Add automated security scanning to the pipeline
5. Regularly update dependencies to patch known vulnerabilities

"""
        
        return md
    
    def generate_pdf_report(self):
        """Generate a PDF report"""
        try:
            # Create a PDF object
            pdf = FPDF()
            pdf.add_page()
            
            # Set font
            pdf.set_font("Arial", "B", 16)
            
            # Title
            pdf.cell(0, 10, "GitHub Repository Analysis Report", 0, 1, "C")
            pdf.ln(10)
            
            # Project Overview
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Project Overview", 0, 1, "L")
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Project Name: {self.repo_analysis['project_name']}", 0, 1, "L")
            pdf.cell(0, 10, f"Primary Purpose: {self.repo_analysis['primary_purpose']}", 0, 1, "L")
            
            # Project Summary
            pdf.multi_cell(0, 10, f"Summary: {self.repo_analysis['project_summary']}")
            pdf.ln(5)
            
            # Vulnerability Summary
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Vulnerability Summary", 0, 1, "L")
            pdf.set_font("Arial", "", 12)
            
            # Count vulnerabilities by severity
            severity_counts = {"High": 0, "Medium": 0, "Low": 0}
            for vuln in self.vulnerabilities:
                severity_counts[vuln['severity']] += 1
            
            pdf.cell(0, 10, f"Total vulnerabilities found: {len(self.vulnerabilities)}", 0, 1, "L")
            pdf.cell(0, 10, f"High severity: {severity_counts['High']}", 0, 1, "L")
            pdf.cell(0, 10, f"Medium severity: {severity_counts['Medium']}", 0, 1, "L")
            pdf.cell(0, 10, f"Low severity: {severity_counts['Low']}", 0, 1, "L")
            pdf.ln(5)
            
            # Top vulnerabilities
            if len(self.vulnerabilities) > 0:
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "Top Vulnerabilities", 0, 1, "L")
                pdf.set_font("Arial", "", 12)
                
                # Sort vulnerabilities by severity
                sorted_vulns = sorted(self.vulnerabilities, key=lambda x: {"High": 0, "Medium": 1, "Low": 2}[x['severity']])
                
                # Display top 5 vulnerabilities
                for i, vuln in enumerate(sorted_vulns[:5]):
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 10, f"{i+1}. {vuln['vulnerability_type']} ({vuln['severity']})", 0, 1, "L")
                    pdf.set_font("Arial", "", 12)
                    pdf.cell(0, 10, f"File: {vuln['file_path']}", 0, 1, "L")
                    pdf.multi_cell(0, 10, f"Description: {vuln['description']}")
                    pdf.multi_cell(0, 10, f"Recommendation: {vuln['recommendation']}")
                    pdf.ln(5)
            
            # Output PDF as bytes
            pdf_output = pdf.output(dest='S').encode('latin1')
            return pdf_output
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            st.error(f"Error generating PDF report: {str(e)}")
            return None
    
    def get_download_link(self, file_content, file_name, link_text):
        """Generate a download link for a file"""
        b64 = base64.b64encode(file_content.encode('utf-8') if isinstance(file_content, str) else file_content).decode()
        return f'<a href="data:file/txt;base64,{b64}" download="{file_name}">{link_text}</a>'


# Setup tab content
with tab1:
    st.header("1. LLM Provider Setup")
    
    # LLM Provider Selection
    llm_option = st.radio("Select LLM Provider", ["Ollama (Local)", "OpenAI API", "Anthropic API"])
    
    if llm_option == "Ollama (Local)":
        st.subheader("Ollama Setup")
        model_name = st.text_input("Model Name", "llama2")
        
        if st.button("Connect to Ollama"):
            with st.spinner("Connecting to Ollama..."):
                st.session_state.llm_provider = OllamaProvider(model_name)
                if st.session_state.llm_provider.client:
                    st.success("Connected to Ollama successfully!")
                    st.session_state.current_step = 2
                else:
                    st.error("Failed to connect to Ollama.")
    
    elif llm_option == "OpenAI API":
        st.subheader("OpenAI API Setup")
        api_key = st.text_input("API Key", type="password")
        model = st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo"])
        
        if st.button("Connect to OpenAI"):
            if api_key:
                with st.spinner("Connecting to OpenAI..."):
                    st.session_state.llm_provider = OpenAIProvider(api_key, model)
                    if st.session_state.llm_provider.client:
                        st.success("Connected to OpenAI successfully!")
                        st.session_state.current_step = 2
                    else:
                        st.error("Failed to connect to OpenAI API.")
            else:
                st.error("Please enter an API key.")
    
    elif llm_option == "Anthropic API":
        st.subheader("Anthropic API Setup")
        api_key = st.text_input("API Key", type="password")
        model = st.selectbox("Model", ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"])
        
        if st.button("Connect to Anthropic"):
            if api_key:
                with st.spinner("Connecting to Anthropic..."):
                    st.session_state.llm_provider = AnthropicProvider(api_key, model)
                    if st.session_state.llm_provider.client:
                        st.success("Connected to Anthropic successfully!")
                        st.session_state.current_step = 2
                    else:
                        st.error("Failed to connect to Anthropic API.")
            else:
                st.error("Please enter an API key.")
    
    st.divider()
    
    # GitHub Repository Setup
    if st.session_state.current_step >= 2:
        st.header("2. GitHub Repository Setup")
        
        repo_url = st.text_input("Repository URL", "https://github.com/username/repository")
        is_private = st.checkbox("Private Repository")
        
        if is_private:
            auth_token = st.text_input("GitHub Personal Access Token", type="password")
        else:
            auth_token = None
        
        if st.button("Clone Repository"):
            if repo_url:
                with st.spinner("Cloning repository..."):
                    repo_handler = RepositoryHandler()
                    repo_path = repo_handler.clone_repository(repo_url, auth_token)
                    
                    if repo_path:
                        st.session_state.repo_path = repo_path
                        st.success(f"Repository cloned successfully!")
                        st.session_state.current_step = 3
                    else:
                        st.error("Failed to clone repository.")
            else:
                st.error("Please enter a repository URL.")

# Analysis tab content
with tab2:
    if st.session_state.current_step >= 3:
        st.header("Repository Analysis")
        
        if st.button("Start Analysis"):
            with st.spinner("Analyzing repository..."):
                # Context Analysis
                context_analyzer = CodeContextAnalyzer(st.session_state.repo_path, st.session_state.llm_provider)
                repo_analysis = context_analyzer.analyze_repository_context()
                st.session_state.repo_analysis = repo_analysis
                
                # Vulnerability Scanning
                vulnerability_scanner = VulnerabilityScanner(st.session_state.repo_path, st.session_state.llm_provider)
                vulnerabilities = vulnerability_scanner.scan_repository()
                st.session_state.vulnerabilities = vulnerabilities
                
                # Generate Pipelines
                pipeline_generator = DeploymentPipelineGenerator(repo_analysis, st.session_state.llm_provider)
                pipelines = pipeline_generator.generate_all_pipelines()
                st.session_state.generated_pipeline = pipelines
                
                # Generate Report
                report_generator = ReportGenerator(repo_analysis, vulnerabilities, pipelines)
                markdown_report = report_generator.generate_markdown_report()
                pdf_report = report_generator.generate_pdf_report()
                
                st.session_state.report_data = {
                    "markdown": markdown_report,
                    "pdf": pdf_report
                }
                
                st.success("Analysis completed successfully!")
                st.session_state.current_step = 4
    else:
        st.info("Please complete the previous steps first.")

# Results tab content
with tab3:
    if st.session_state.current_step >= 4 and st.session_state.repo_analysis and st.session_state.vulnerabilities:
        st.header("Analysis Results")
        
        # Repository Context
        st.subheader("Repository Context")
        with st.expander("Project Overview", expanded=True):
            st.markdown(f"**Project Name:** {st.session_state.repo_analysis['project_name']}")
            st.markdown(f"**Project Summary:** {st.session_state.repo_analysis['project_summary']}")
            st.markdown(f"**Primary Purpose:** {st.session_state.repo_analysis['primary_purpose']}")
            st.markdown(f"**Architecture:** {st.session_state.repo_analysis['architecture']}")
        
        with st.expander("Technologies Used"):
            for tech in st.session_state.repo_analysis['technologies']:
                st.markdown(f"- {tech}")
        
        with st.expander("Key Components"):
            for component in st.session_state.repo_analysis['key_components']:
                st.markdown(f"**{component['name']}:** {component['function']}")
        
        # Vulnerability Analysis
        st.subheader("Vulnerability Analysis")
        
        # Count vulnerabilities by severity
        severity_counts = {"High": 0, "Medium": 0, "Low": 0}
        for vuln in st.session_state.vulnerabilities:
            severity_counts[vuln['severity']] += 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("High Severity", severity_counts['High'], delta=None, delta_color="inverse")
        with col2:
            st.metric("Medium Severity", severity_counts['Medium'], delta=None, delta_color="inverse")
        with col3:
            st.metric("Low Severity", severity_counts['Low'], delta=None, delta_color="inverse")
        
        # Create a DataFrame for the vulnerabilities
        if len(st.session_state.vulnerabilities) > 0:
            vuln_data = []
            for vuln in st.session_state.vulnerabilities:
                line_numbers = ', '.join([str(ln) for ln in vuln['line_numbers']]) if isinstance(vuln['line_numbers'], list) else vuln['line_numbers']
                vuln_data.append({
                    "Type": vuln['vulnerability_type'],
                    "Severity": vuln['severity'],
                    "File": vuln['file_path'],
                    "Line(s)": line_numbers,
                    "Description": vuln['description'],
                    "Recommendation": vuln['recommendation']
                })
            
            df = pd.DataFrame(vuln_data)
            
            # Add color to severity column
            def highlight_severity(val):
                if val == "High":
                    return 'background-color: #ff4b4b; color: white'
                elif val == "Medium":
                    return 'background-color: #ffa500; color: white'
                elif val == "Low":
                    return 'background-color: #4b96ff; color: white'
                return ''
            
            # Display the DataFrame
            st.dataframe(df.style.applymap(highlight_severity, subset=['Severity']))
            
            # Display detailed vulnerabilities
            with st.expander("Detailed Vulnerabilities"):
                for i, vuln in enumerate(st.session_state.vulnerabilities):
                    severity_color = {
                        "High": "vulnerability-high",
                        "Medium": "vulnerability-medium",
                        "Low": "vulnerability-low"
                    }
                    
                    st.markdown(f"### {i+1}. <span class='{severity_color[vuln['severity']]}'>{vuln['vulnerability_type']} ({vuln['severity']})</span>", unsafe_allow_html=True)
                    st.markdown(f"**File:** `{vuln['file_path']}`")
                    
                    line_numbers = ', '.join([str(ln) for ln in vuln['line_numbers']]) if isinstance(vuln['line_numbers'], list) else vuln['line_numbers']
                    st.markdown(f"**Line(s):** {line_numbers}")
                    
                    st.markdown(f"**Description:** {vuln['description']}")
                    st.markdown(f"**Recommendation:** {vuln['recommendation']}")
                    st.divider()
        else:
            st.info("No vulnerabilities found.")
        
        # Generate visualizations
        if len(st.session_state.vulnerabilities) > 0:
            st.subheader("Vulnerability Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pie chart of vulnerabilities by severity
                fig1, ax1 = plt.subplots()
                labels = ['High', 'Medium', 'Low']
                sizes = [severity_counts['High'], severity_counts['Medium'], severity_counts['Low']]
                colors = ['#ff4b4b', '#ffa500', '#4b96ff']
                explode = (0.1, 0, 0)
                
                ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
                ax1.axis('equal')
                st.pyplot(fig1)
            
            with col2:
                # Bar chart of vulnerability types
                vuln_types = {}
                for vuln in st.session_state.vulnerabilities:
                    vuln_type = vuln['vulnerability_type']
                    if vuln_type in vuln_types:
                        vuln_types[vuln_type] += 1
                    else:
                        vuln_types[vuln_type] = 1
                
                fig2, ax2 = plt.subplots()
                y_pos = range(len(vuln_types))
                ax2.barh(y_pos, list(vuln_types.values()), align='center')
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(list(vuln_types.keys()))
                ax2.invert_yaxis()
                ax2.set_xlabel('Count')
                ax2.set_title('Vulnerability Types')
                st.pyplot(fig2)
        
        # Download Reports
        st.subheader("Download Reports")
        
        if 'report_data' in st.session_state and st.session_state.report_data:
            col1, col2 = st.columns(2)
            
            with col1:
                report_generator = ReportGenerator(st.session_state.repo_analysis, st.session_state.vulnerabilities, st.session_state.generated_pipeline)
                markdown_link = report_generator.get_download_link(
                    st.session_state.report_data['markdown'],
                    f"{st.session_state.repo_analysis['project_name']}_report.md",
                    "Download Markdown Report"
                )
                st.markdown(markdown_link, unsafe_allow_html=True)
            
            with col2:
                if st.session_state.report_data['pdf']:
                    pdf_link = report_generator.get_download_link(
                        st.session_state.report_data['pdf'],
                        f"{st.session_state.repo_analysis['project_name']}_report.pdf",
                        "Download PDF Report"
                    )
                    st.markdown(pdf_link, unsafe_allow_html=True)
                else:
                    st.error("PDF report generation failed.")
    else:
        st.info("Please complete the analysis first.")

# Deployment tab content
with tab4:
    if st.session_state.current_step >= 4 and st.session_state.generated_pipeline:
        st.header("Deployment Pipeline")
        
        # Pipeline selection
        pipeline_type = st.selectbox("Select Pipeline Type", ["GitHub Actions", "GitLab CI", "Jenkins", "Dockerfile"])
        
        if pipeline_type == "GitHub Actions":
            st.code(st.session_state.generated_pipeline['github_actions'], language="yaml")
            
            # Download pipeline
            pipeline_content = st.session_state.generated_pipeline['github_actions']
            b64 = base64.b64encode(pipeline_content.encode()).decode()
            href = f'<a href="data:text/yaml;base64,{b64}" download="github-actions-workflow.yml">Download GitHub Actions Workflow</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        elif pipeline_type == "GitLab CI":
            st.code(st.session_state.generated_pipeline['gitlab_ci'], language="yaml")
            
            # Download pipeline
            pipeline_content = st.session_state.generated_pipeline['gitlab_ci']
            b64 = base64.b64encode(pipeline_content.encode()).decode()
            href = f'<a href="data:text/yaml;base64,{b64}" download="gitlab-ci.yml">Download GitLab CI Pipeline</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        elif pipeline_type == "Jenkins":
            st.code(st.session_state.generated_pipeline['jenkins'], language="groovy")
            
            # Download pipeline
            pipeline_content = st.session_state.generated_pipeline['jenkins']
            b64 = base64.b64encode(pipeline_content.encode()).decode()
            href = f'<a href="data:text/plain;base64,{b64}" download="Jenkinsfile">Download Jenkinsfile</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        elif pipeline_type == "Dockerfile":
            st.code(st.session_state.generated_pipeline['dockerfile'], language="dockerfile")
            
            # Download pipeline
            pipeline_content = st.session_state.generated_pipeline['dockerfile']
            b64 = base64.b64encode(pipeline_content.encode()).decode()
            href = f'<a href="data:text/plain;base64,{b64}" download="Dockerfile">Download Dockerfile</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        # Deployment instructions
        st.subheader("Deployment Instructions")
        
        with st.expander("How to use this pipeline"):
            if pipeline_type == "GitHub Actions":
                st.markdown("""
                ### GitHub Actions Setup Instructions
                
                1. Create a `.github/workflows` directory in your repository if it doesn't exist
                2. Save the generated workflow file as `.github/workflows/main.yml`
                3. Push the changes to your GitHub repository
                4. Configure the required secrets in your GitHub repository settings:
                   - Go to Settings > Secrets and Variables > Actions
                   - Add the necessary secrets referenced in the workflow file
                
                The workflow will automatically run on push to the main branch and on pull requests.
                """)
            
            elif pipeline_type == "GitLab CI":
                st.markdown("""
                ### GitLab CI Setup Instructions
                
                1. Save the generated pipeline configuration as `.gitlab-ci.yml` in the root of your repository
                2. Push the changes to your GitLab repository
                3. Configure the required variables in your GitLab repository settings:
                   - Go to Settings > CI/CD > Variables
                   - Add the necessary variables referenced in the pipeline file
                
                The pipeline will automatically run on push to the main branch and on merge requests.
                """)
            
            elif pipeline_type == "Jenkins":
                st.markdown("""
                ### Jenkins Setup Instructions
                
                1. Save the generated pipeline as `Jenkinsfile` in the root of your repository
                2. Configure a Jenkins pipeline job:
                   - Create a new Pipeline job
                   - Select "Pipeline script from SCM" as the definition
                   - Configure your repository URL and credentials
                   - Set the script path to "Jenkinsfile"
                3. Configure the required credentials in Jenkins:
                   - Go to Manage Jenkins > Manage Credentials
                   - Add the necessary credentials referenced in the Jenkinsfile
                
                The pipeline will run according to the triggers configured in the Jenkinsfile.
                """)
            
            elif pipeline_type == "Dockerfile":
                st.markdown("""
                ### Docker Setup Instructions
                
                1. Save the generated Dockerfile as `Dockerfile` in the root of your repository
                2. Build the Docker image:
                   ```
                   docker build -t your-app-name .
                   ```
                3. Run the Docker container:
                   ```
                   docker run -p 8501:8501 your-app-name
                   ```
                
                This Dockerfile can be used in conjunction with any of the CI/CD pipelines to build and deploy your application.
                """)
        
        # Cloud deployment options
        st.subheader("Cloud Deployment Options")
        
        cloud_provider = st.selectbox("Select Cloud Provider", ["AWS", "Azure", "Google Cloud"])
        
        with st.expander("Deployment Instructions"):
            if cloud_provider == "AWS":
                st.markdown("""
                ### AWS Deployment Instructions
                
                #### Option 1: AWS App Runner
                
                1. Push your Docker image to Amazon ECR:
                   ```
                   aws ecr create-repository --repository-name your-app-name
                   aws ecr get-login-password | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
                   docker tag your-app-name:latest <account-id>.dkr.ecr.<region>.amazonaws.com/your-app-name:latest
                   docker push <account-id>.dkr.ecr.<region>.amazonaws.com/your-app-name:latest
                   ```
                
                2. Create an AWS App Runner service:
                   - Go to AWS App Runner console
                   - Create a new service
                   - Select "Container registry" as source
                   - Choose "Amazon ECR" as provider
                   - Select your repository and image
                   - Configure service settings and deploy
                
                #### Option 2: AWS Elastic Beanstalk
                
                1. Install the EB CLI:
                   ```
                   pip install awsebcli
                   ```
                
                2. Initialize EB CLI:
                   ```
                   eb init -p docker your-app-name
                   ```
                
                3. Deploy your application:
                   ```
                   eb create your-environment-name
                   ```
                
                #### Option 3: AWS ECS/Fargate
                
                1. Push your Docker image to Amazon ECR (as shown above)
                2. Create an ECS cluster (if none exists)
                3. Define an ECS task definition
                4. Create an ECS service using Fargate launch type
                5. Configure load balancing and networking as needed
                """)
            
            elif cloud_provider == "Azure":
                st.markdown("""
                ### Azure Deployment Instructions
                
                #### Option 1: Azure App Service
                
                1. Create an Azure Container Registry (ACR):
                   ```
                   az acr create --resource-group your-resource-group --name yourAcrName --sku Basic
                   ```
                
                2. Build and push your image to ACR:
                   ```
                   az acr login --name yourAcrName
                   az acr build --registry yourAcrName --image your-app-name:latest .
                   ```
                
                3. Create an App Service plan:
                   ```
                   az appservice plan create --name your-plan-name --resource-group your-resource-group --sku B1 --is-linux
                   ```
                
                4. Create and deploy to Web App:
                   ```
                   az webapp create --resource-group your-resource-group --plan your-plan-name --name your-app-name --deployment-container-image-name yourAcrName.azurecr.io/your-app-name:latest
                   ```
                
                #### Option 2: Azure Container Instances
                
                1. Push your image to ACR (as shown above)
                2. Deploy to Container Instances:
                   ```
                   az container create --resource-group your-resource-group --name your-container-name --image yourAcrName.azurecr.io/your-app-name:latest --dns-name-label your-dns-label --ports 8501
                   ```
                
                #### Option 3: Azure Kubernetes Service
                
                1. Create an AKS cluster:
                   ```
                   az aks create --resource-group your-resource-group --name your-aks-cluster --node-count 1 --enable-addons monitoring --generate-ssh-keys
                   ```
                
                2. Connect to the cluster:
                   ```
                   az aks get-credentials --resource-group your-resource-group --name your-aks-cluster
                   ```
                
                3. Deploy your application using kubectl
                """)
            
            elif cloud_provider == "Google Cloud":
                st.markdown("""
                ### Google Cloud Deployment Instructions
                
                #### Option 1: Google Cloud Run
                
                1. Build and push your Docker image to Google Container Registry:
                   ```
                   gcloud auth configure-docker
                   docker build -t gcr.io/your-project-id/your-app-name:latest .
                   docker push gcr.io/your-project-id/your-app-name:latest
                   ```
                
                2. Deploy to Cloud Run:
                   ```
                   gcloud run deploy your-app-name --image gcr.io/your-project-id/your-app-name:latest --platform managed --region us-central1 --allow-unauthenticated
                   ```
                
                #### Option 2: Google App Engine
                
                1. Create an app.yaml file:
                   ```yaml
                   runtime: custom
                   env: flex
                   ```
                
                2. Deploy to App Engine:
                   ```
                   gcloud app deploy
                   ```
                
                #### Option 3: Google Kubernetes Engine
                
                1. Create a GKE cluster:
                   ```
                   gcloud container clusters create your-cluster-name --zone us-central1-a --num-nodes=1
                   ```
                
                2. Configure kubectl:
                   ```
                   gcloud container clusters get-credentials your-cluster-name --zone us-central1-a
                   ```
                
                3. Deploy your application using kubectl
                """)

    else:
        st.info("Please complete the analysis first.")


if __name__ == "__main__":
    # This code will only run when the script is executed directly
    pass
