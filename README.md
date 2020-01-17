# Azurlane-Auto
Renovate and Rewrite base on [ALAuto](https://github.com/Egoistically/ALAuto).  
**Thanks so much. Egoistically**

**Only support EN Server.**

## Requirements on Windows
* Python 3.7.X installed and added to PATH.
* Latest [ADB](https://developer.android.com/studio/releases/platform-tools) added to PATH.
* Install [Tesseract](https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.0.0-alpha.20191030.exe) added to PATH
* ADB debugging enabled emulator with **1920x1080 resolution**
* **Recommend and Testing on Bluestack 4** 
* **note : "Tesseract" use for convert image to text**

## Installation and Usage
1. Download from release.
2. Install the required packages via `pip3` with the command `pip3 install -r requirements.txt`.
3. Enable adb debugging on your emulator, on Nox you might also need to enable root.
4. Change config.ini's IP:PORT to 127.0.0.1 and your emulator's adb port, then change the rest to your likings. If you are using your own phone/device for the bot, enable debbuging on your device and change IP:PORT to PHONE.
5. Run `ALAuto` using the command `python ALAuto.py`.

Check the [Wiki](https://github.com/Egoistically/ALAuto/wiki/Config.ini-and-Modules-explanation) for more information. 

**Thanks so much again. Egoistically**
