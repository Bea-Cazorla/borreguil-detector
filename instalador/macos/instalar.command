#!/bin/bash
# Instalador de Borreguil · Wet Meadow Finder para macOS (27-11-2026: como los de MonitorEO y eHabitat+). Se pega en
# la app Terminal:
#
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/monitoreo-obsnev/borreguil-detector/main/instalador/macos/instalar.command)"
#
# (lo que baja curl no queda en cuarentena, así que macOS no lo bloquea), o se abre con doble clic desde el zip de la
# Release, Borreguil-macos.zip, donde se llama «Instalar Borreguil.command» (sin firma: la primera vez macOS pide
# permiso en Ajustes del Sistema → Privacidad y seguridad).
#
# Instala en ~/Borreguil (la carpeta de los datos de la app), sin permisos de administrador:
#   - micromamba (el instalador de paquetes de conda-forge), en ~/Borreguil/micromamba;
#   - el entorno de Python de la app (Python 3.12 y las librerías de environment.yml), en ~/Borreguil/entorno;
#   - el código de la app, en ~/Borreguil/programa: el publicado (el commit de codigo.json, el mismo que los programas
#     instalados descargan al abrirse) o, si el instalador se abre desde una copia del repositorio, el de esa copia.
# Pone en el Escritorio la app «Borreguil», que abre la app en su ventana sin la Terminal (lo que escribe va a
# ~/Borreguil/salida.log), y la abre. No lleva firma, pero la crea este ordenador: no queda en cuarentena y macOS la
# abre sin avisos. Al abrirse, la app se actualiza sola (actualizador.py).
# Si el entorno ya está, se hizo con la misma lista de librerías (environment.yml), las importa todas y pasa el
# autodiagnóstico del programa con el código nuevo, no se vuelve a instalar: solo se ponen al día el código y la app
# del Escritorio. Si no, se instala de nuevo. Para forzarlo, borra ~/Borreguil/entorno o abre el instalador con
# BORREGUIL_REINSTALAR=1.
# Otras variables: BORREGUIL_HOME (carpeta de la instalación y de los datos), BORREGUIL_ESCRITORIO (carpeta donde va la
# app) y BORREGUIL_NO_ABRIR=1 (no abrir la app al final).
# Todo va dentro de main, que se llama al final: una descarga cortada no ejecuta nada.

main() {
	set -euo pipefail
	# Lo que la cuenta del usuario pueda tener puesto para su propio Python o pip no cuenta aquí (pip instalaría
	# pywebview fuera del entorno, PYTHONPATH haría fallar el autodiagnóstico de un entorno bueno...).
	unset PYTHONPATH PYTHONHOME CONDA_SUBDIR PIP_USER PIP_TARGET PIP_PREFIX PIP_REQUIRE_VIRTUALENV
	REPO=monitoreo-obsnev/borreguil-detector
	RAW="https://raw.githubusercontent.com/$REPO/main"
	DIR="${BORREGUIL_HOME:-$HOME/Borreguil}"
	ENV="$DIR/entorno"
	LISTA_ENV="$ENV/.borreguil-environment.yml"  # la lista de librerías con la que se hizo el entorno
	MAMBA_DIR="$DIR/micromamba"
	MAMBA="$MAMBA_DIR/micromamba"
	CODE="$DIR/programa"
	NEW="$DIR/programa.nuevo"
	DESKTOP="${BORREGUIL_ESCRITORIO:-$HOME/Desktop}"
	APP="$DESKTOP/Borreguil.app"

	fail() {  # fail MENSAJE [LÍNEAS DE QUÉ HACER]: por defecto, la conexión, la causa de casi todos los errores
		echo
		echo "ERROR: $1"
		if [ $# -gt 1 ]; then
			printf '%s\n' "${@:2}"
		else
			echo "Comprueba la conexión a internet y vuelve a abrir el instalador. Lo de arriba dice qué ha fallado."
			echo "Si sigue fallando, avisa en https://github.com/$REPO/issues"
		fi
		read -r -p "Pulsa Intro para cerrar. " _ < /dev/tty || true
		exit 1
	}
	trap 'fail "la instalación se ha parado (lo de arriba dice dónde)."' ERR

	echo "Instalador de Borreguil · Wet Meadow Finder"
	echo "==========================================="
	echo
	echo "Instala en $DIR, sin permisos de administrador, lo que necesita la app: Python y sus librerías"
	echo "científicas (unos 0,3 GB de descarga y 1,4 GB en disco, solo la primera vez). Al terminar, la app"
	echo "«Borreguil» queda en el Escritorio."
	echo

	case "$(uname -m)" in
		arm64) PLATFORM=osx-arm64 ;;
		x86_64) PLATFORM=osx-64 ;;
		*) fail "este Mac no es ni Apple Silicon ni Intel ($(uname -m))." ;;
	esac
	# Una Terminal abierta con Rosetta dice x86_64 en un Mac Apple Silicon: el entorno, el del procesador de verdad.
	if [ "$(sysctl -in sysctl.proc_translated 2>/dev/null || true)" = 1 ]; then
		PLATFORM=osx-arm64
	fi

	# La app va en el Escritorio, y macOS pregunta (una vez) si la Terminal puede entrar en él. Si se le dijo que no,
	# todo falla con «Operation not permitted» y macOS no vuelve a preguntar: se dice ahora, antes de las descargas.
	TEST="$DESKTOP/.borreguil-instalador-prueba"
	if ! { mkdir -p "$DESKTOP" && touch "$TEST" && rm -f "$TEST"; } 2>/dev/null; then
		fail "la Terminal no tiene permiso para entrar en el Escritorio, donde el instalador deja la app." \
			"Ve a Ajustes del Sistema → Privacidad y seguridad → Archivos y carpetas → Terminal, activa Escritorio" \
			"y vuelve a abrir el instalador."
	fi
	mkdir -p "$DIR"
	# El código se cambia por el nuevo: con la app abierta, dejaría de funcionar a medias. El servidor de la app es otro
	# proceso (launcher.py --servidor); si la app se cerró a la fuerza puede quedar solo, sin ventana: se cierra aquí.
	ABIERTA="" SERVIDORES=""
	for PID in $(pgrep -f "$CODE/launcher.py" 2>/dev/null || true); do
		if ps -o args= -p "$PID" 2>/dev/null | grep -q -- '--servidor'; then
			SERVIDORES="$SERVIDORES $PID"
		else
			ABIERTA="$ABIERTA $PID"
		fi
	done
	if [ -n "$ABIERTA" ]; then
		fail "Borreguil está abierto." "Cierra la ventana de la app y vuelve a abrir el instalador. Si no ves ninguna," \
			"reinicia el Mac y vuelve a abrirlo."
	fi
	if [ -n "$SERVIDORES" ]; then
		echo "Cerrando el servidor de una Borreguil que se quedó abierto..."
		kill $SERVIDORES 2>/dev/null || true
		sleep 3
		kill -9 $SERVIDORES 2>/dev/null || true
	fi

	# El código: el de la copia del repositorio que contiene este instalador o, si no, el publicado.
	HERE=""
	if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
		HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
	fi
	rm -rf "$NEW"
	LOCAL=""
	if [ -n "$HERE" ] && [ -f "$HERE/../../launcher.py" ] && [ -f "$HERE/../../environment.yml" ]; then
		LOCAL="$(cd "$HERE/../.." && pwd -P)"
		# el instalador que viene dentro del código instalado (~/Borreguil/programa) instala el publicado
		if [ -d "$CODE" ] && [ "$LOCAL" = "$(cd "$CODE" && pwd -P)" ]; then
			LOCAL=""
		fi
	fi
	if [ -n "$LOCAL" ]; then
		echo "Código de la app: el de la copia del repositorio en $LOCAL"
		mkdir -p "$NEW"
		# sin lo que el repositorio no guarda (.gitignore: salidas, compilaciones, claves...)
		IGNORA=()
		[ -f "$LOCAL/.gitignore" ] && IGNORA=(--exclude-from "$LOCAL/.gitignore")
		rsync -a --exclude /.git --exclude /.serena ${IGNORA[@]+"${IGNORA[@]}"} "$LOCAL/" "$NEW/" \
			|| fail "no se ha podido copiar el código de $LOCAL."
	else
		echo "Buscando la versión publicada de la app (github.com/$REPO)..."
		LISTA=$(curl -fsSL "$RAW/codigo.json") || fail "no se ha podido leer qué versión de la app está publicada."
		COMMIT=$(printf '%s\n' "$LISTA" | sed -n 's/^[[:space:]]*"commit":[[:space:]]*"\([0-9a-f]\{7,40\}\)".*/\1/p' | head -n 1 || true)
		VERSION=$(printf '%s\n' "$LISTA" | sed -n 's/^[[:space:]]*"version":[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1 || true)
		[ -n "$COMMIT" ] || fail "la lista del código publicado (codigo.json) no dice de qué commit sale."
		echo "Descargando el código de la versión ${VERSION:-?} (commit ${COMMIT:0:7})..."
		TMP="$DIR/descarga"
		rm -rf "$TMP" && mkdir -p "$TMP"
		curl -fL --progress-bar -o "$TMP/codigo.zip" "https://github.com/$REPO/archive/$COMMIT.zip" \
			|| fail "no se ha podido descargar el código de la app."
		(cd "$TMP" && unzip -q codigo.zip) || fail "el código descargado está dañado."
		SRC="$(find "$TMP" -mindepth 1 -maxdepth 1 -type d | head -n 1 || true)"
		[ -n "$SRC" ] && [ -f "$SRC/launcher.py" ] && [ -f "$SRC/app.py" ] \
			|| fail "el código descargado no trae la app (launcher.py y app.py)."
		mv "$SRC" "$NEW"
		rm -rf "$TMP"
	fi
	# La lista de librerías va con el código; un código publicado antes de que la hubiera usa la de main.
	if [ ! -f "$NEW/environment.yml" ]; then
		curl -fsSL -o "$NEW/environment.yml" "$RAW/environment.yml" \
			|| fail "no se ha podido descargar la lista de librerías (environment.yml)."
	fi

	# Lo que se comprueba del entorno: que importa todas las librerías de environment.yml (por su nombre de módulo;
	# tests/test_instalador_mac.py comprueba que no falta ninguna) y el autodiagnóstico del programa (el mismo que pasa
	# la compilación: los módulos de la app y un modelo que carga y predice con la scikit-learn del entorno).
	MODULOS="numpy,pandas,scipy,sklearn,skimage,geopandas,shapely,pyproj,rasterio,tifffile,requests,PIL,openpyxl"
	MODULOS="$MODULOS,folium,streamlit,streamlit_folium,altair,pystac_client,planetary_computer,ee,joblib,docx,webview"
	librerias() {
		PYTHONNOUSERSITE=1 "$ENV/bin/python" -s -c "import $MODULOS; print('Librerías: todas se importan.')"
	}
	selftest() {
		PYTHONNOUSERSITE=1 "$ENV/bin/python" -s "$NEW/launcher.py" --selftest
	}
	misma_lista() {  # el entorno se hizo con esta lista (sin contar los comentarios)
		[ -f "$LISTA_ENV" ] && [ "$(sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$LISTA_ENV")" \
			= "$(sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$NEW/environment.yml")" ]
	}

	if [ "${BORREGUIL_REINSTALAR:-0}" != "1" ] && [ -x "$ENV/bin/python" ] && misma_lista \
		&& librerias >/dev/null 2>&1 && selftest >/dev/null 2>&1; then
		echo "El entorno de Python ya está instalado en $ENV."
	else
		if [ -d "$ENV" ]; then
			echo "El entorno de Python de $ENV no está completo o es de otra lista de librerías: se instala de nuevo."
		fi
		mkdir -p "$MAMBA_DIR"
		# un micromamba que no funciona en este Mac (p. ej. el de Intel, pasado a un Apple Silicon sin Rosetta)
		if [ -x "$MAMBA" ] && ! "$MAMBA" --version >/dev/null 2>&1; then
			rm -f "$MAMBA"
		fi
		if [ ! -x "$MAMBA" ]; then
			echo "1/3 Descargando micromamba (el instalador de paquetes de conda-forge)..."
			URL="https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-$PLATFORM"
			curl -fL --progress-bar -o "$MAMBA.part" "$URL" || fail "no se ha podido descargar micromamba."
			SHA=$(curl -fsL "$URL.sha256") || fail "no se ha podido descargar la huella de micromamba."
			[ "$(shasum -a 256 "$MAMBA.part" | cut -d' ' -f1)" = "$SHA" ] || fail "la descarga de micromamba está dañada."
			chmod +x "$MAMBA.part"
			mv "$MAMBA.part" "$MAMBA"
		else
			echo "1/3 micromamba ya está en $MAMBA_DIR."
		fi

		echo "2/3 Lista de librerías: environment.yml"
		echo "3/3 Instalando Python y las librerías de la app (unos minutos)..."
		rm -rf "$ENV"
		export MAMBA_ROOT_PREFIX="$MAMBA_DIR" PYTHONNOUSERSITE=1
		"$MAMBA" create -y -p "$ENV" -f "$NEW/environment.yml" \
			|| fail "no se ha podido instalar el entorno de Python."
		"$MAMBA" clean -y --all >/dev/null 2>&1 || true  # la caché de los paquetes, ya instalados
		echo
		echo "Comprobando la instalación..."
		librerias || fail "el entorno se ha instalado, pero falta alguna librería (lo de arriba dice cuál)."
		selftest || fail "el entorno se ha instalado, pero el autodiagnóstico de la app falla (lo de arriba dice qué)."
		cp "$NEW/environment.yml" "$LISTA_ENV"  # solo cuando el entorno ya está bien
	fi

	# El código nuevo sustituye al de antes solo cuando el entorno ya le vale.
	rm -rf "$CODE"
	mv "$NEW" "$CODE"

	# La app del Escritorio: un paquete .app con un guion que abre Borreguil con el Python del entorno.
	echo "Creando la app «Borreguil» en $DESKTOP..."
	rm -rf "$APP"
	mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
	# Su icono, la foto de un borreguil de la app recortada en cuadrado (sin ella, el genérico de las apps).
	FOTO="$CODE/imagenes/borreguil_cerro_del_caballo.jpg"
	if [ -f "$FOTO" ]; then
		SET="$DIR/icono.iconset"
		rm -rf "$SET" && mkdir -p "$SET"
		LADO=$(sips -g pixelWidth -g pixelHeight "$FOTO" 2>/dev/null | awk '/pixel(Width|Height):/ {print $2}' | sort -n | head -n 1 || true)
		if [ -n "$LADO" ] && sips -s format png -c "$LADO" "$LADO" "$FOTO" --out "$SET/cuadrado.png" >/dev/null 2>&1; then
			for s in 16 32 128 256 512; do
				sips -z $s $s "$SET/cuadrado.png" --out "$SET/icon_${s}x${s}.png" >/dev/null 2>&1 || true
				sips -z $((s * 2)) $((s * 2)) "$SET/cuadrado.png" --out "$SET/icon_${s}x${s}@2x.png" >/dev/null 2>&1 || true
			done
			rm -f "$SET/cuadrado.png"
			iconutil -c icns "$SET" -o "$APP/Contents/Resources/borreguil.icns" >/dev/null 2>&1 || true
		fi
		rm -rf "$SET"
	fi
	cat > "$APP/Contents/Info.plist" <<-EOF
	<?xml version="1.0" encoding="UTF-8"?>
	<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
	<plist version="1.0">
	<dict>
		<key>CFBundleName</key><string>Borreguil</string>
		<key>CFBundleDisplayName</key><string>Borreguil</string>
		<key>CFBundleIdentifier</key><string>es.ugr.borreguil</string>
		<key>CFBundleExecutable</key><string>borreguil</string>
		<key>CFBundleIconFile</key><string>borreguil</string>
		<key>CFBundlePackageType</key><string>APPL</string>
		<key>CFBundleVersion</key><string>1</string>
		<key>NSHighResolutionCapable</key><true/>
	</dict>
	</plist>
	EOF
	{
		echo '#!/bin/bash'
		echo '# Abre Borreguil en su ventana (lo hizo su instalador). Lo que escribe va a salida.log, en la carpeta Borreguil.'
		printf 'DIR=%q\nENV=%q\nCODE=%q\n' "$DIR" "$ENV" "$CODE"
		cat <<-'EOF'
		SALIDA="$DIR/salida.log"
		if [ -f "$SALIDA" ] && [ "$(stat -f %z "$SALIDA")" -gt 5000000 ]; then
			mv -f "$SALIDA" "$SALIDA.anterior"  # que no crezca sin fin
		fi
		unset PYTHONPATH PYTHONHOME
		export PYTHONNOUSERSITE=1 BORREGUIL_DATA_DIR="$DIR" BORREGUIL_SALIDA="$SALIDA"
		exec "$ENV/bin/python" -s "$CODE/launcher.py" "$@" >> "$SALIDA" 2>&1
		EOF
	} > "$APP/Contents/MacOS/borreguil"
	chmod +x "$APP/Contents/MacOS/borreguil"

	echo
	echo "Hecho. La app se abre ahora en su ventana; para cerrarla, cierra esa ventana."
	echo "Para volver a abrirla: doble clic en «Borreguil», en el Escritorio. Ya puedes cerrar la Terminal."
	if [ "${BORREGUIL_NO_ABRIR:-0}" != "1" ]; then
		open "$APP"
	fi
}

main "$@"
