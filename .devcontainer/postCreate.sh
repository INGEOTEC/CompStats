#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update -qq

# Librerías de sistema que requiere el Chrome headless de quarto (chrome-headless-shell)
# para poder arrancar. Sin ellas, quarto falla al rasterizar los diagramas mermaid con
# "error while loading shared libraries: libatk-1.0.so.0: cannot open shared object file".
sudo apt-get install -y --no-install-recommends \
	libatk1.0-0t64 \
	libatk-bridge2.0-0t64 \
	libatspi2.0-0t64 \
	libdbus-1-3 \
	libxcomposite1 \
	libxdamage1 \
	libxfixes3 \
	libxrandr2 \
	libgbm1 \
	libxkbcommon0 \
	libasound2t64 \
	fonts-liberation

# Instalar GitHub CLI (gh) desde el repositorio oficial para tener la versión más reciente
(type -p wget >/dev/null || (sudo apt update && sudo apt install wget -y)) \
	&& sudo mkdir -p -m 755 /etc/apt/keyrings \
	&& out=$(mktemp) && wget -nv -O"$out" https://cli.github.com/packages/githubcli-archive-keyring.gpg \
	&& cat "$out" | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& sudo mkdir -p -m 755 /etc/apt/sources.list.d \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& sudo apt update \
	&& sudo apt install gh -y

# Instalar uv y usarlo para instalar los paquetes con el Python del sistema
# del contenedor (--system), sin crear un virtualenv: todo ya corre aislado
# dentro del propio docker, así que un venv sería una capa extra innecesaria.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# La imagen base deja site-packages y /usr/local/bin como propiedad de root,
# pero el devcontainer corre como el usuario "vscode": sin esto, uv falla
# con "Permission denied" al instalar en el Python del sistema.
sudo chown -R "$(id -u):$(id -g)" \
	"$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
	/usr/local/bin /usr/local/share /usr/local/etc

bash .devcontainer/python.sh
