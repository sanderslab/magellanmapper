; MagellanMapper Installer Script

; Based on NSIS example script:
; NSIS Modern User Interface
; Welcome/Finish Page Example Script
; Written by Joost Verburg



;--------------------------------
; use Modern UI

  !include "MUI2.nsh"
  !include x64.nsh
  !include LogicLib.nsh
  !include StrFunc.nsh

;--------------------------------
; set up defaults
  ; should pass version as `/DVER=x.y.z`
  !ifndef VER
    !define VER ""
  !endif
  !define BASEDIR "..\..\dist\win"
  !define APP_NAME "MagellanMapper"
  !define APP_NAME_VER "${APP_NAME}-${VER}"

  ; application name and output installer file
  Name "${APP_NAME} ${VER}"
  OutFile "${BASEDIR}\${APP_NAME_VER}-win-installer.exe"

  ; default installation folder, overriding with registry key if available
  InstallDir "$LOCALAPPDATA\${APP_NAME}\${APP_NAME_VER}"
  InstallDirRegKey HKCU "Software\${APP_NAME_VER}" ""
  
  ; use standard privileges
  RequestExecutionLevel user
  
  Var StartMenuFolder
  
  ; prep string location function
  ${StrLoc}

;--------------------------------
; set up interface settings

  !define MUI_ABORTWARNING



;--------------------------------
; set up pages

  !define MUI_WELCOMEPAGE_TEXT "Welcome to ${APP_NAME}, a graphical imaging informatics suite for 3D reconstruction and automated analysis of whole specimens and atlases."
  !insertmacro MUI_PAGE_WELCOME
  !insertmacro MUI_PAGE_LICENSE "..\LICENSE.txt"
  !insertmacro MUI_PAGE_COMPONENTS
  
  
  ; install directory selection
  !insertmacro MUI_PAGE_DIRECTORY
  
  ; ensure that the selected install directory has app folder name to avoid
  ; installing directly into general purpose folders such as Documents
  ; or Program Files (though no UAC privileges are elevated) 
  !define MUI_PAGE_CUSTOMFUNCTION_PRE instFilesPre
  Function instFilesPre
    ${If} ${FileExists} "$INSTDIR\*"
      ${StrLoc} $0 $INSTDIR "${APP_NAME}" ">"
      ${If} $0 == ""
        StrCpy $INSTDIR "$INSTDIR\${APP_NAME_VER}"
      ${endIf}
    ${endIf}
  FunctionEnd
  
  
  ; Start Menu folder configuration
  !define MUI_STARTMENUPAGE_REGISTRY_ROOT "HKCU" 
  !define MUI_STARTMENUPAGE_REGISTRY_KEY "Software\${APP_NAME_VER}" 
  !define MUI_STARTMENUPAGE_REGISTRY_VALUENAME "Start Menu Folder"

  !insertmacro MUI_PAGE_STARTMENU Application $StartMenuFolder
  
  ; install files page
  !insertmacro MUI_PAGE_INSTFILES
  
  ; finish page with option to launch app
  !define MUI_FINISHPAGE_TEXT "${APP_NAME} has been installed and can be launched from $INSTDIR\${APP_NAME}.exe or from the Start Menu."
  !define MUI_FINISHPAGE_RUN "$INSTDIR/${APP_NAME}.exe"
  !define MUI_FINISHPAGE_LINK "${APP_NAME} website"
  !define MUI_FINISHPAGE_LINK_LOCATION https://github.com/sanderslab/magellanmapper
  !insertmacro MUI_PAGE_FINISH


  ; uninstaller page
  !define MUI_WELCOMEPAGE_TEXT "Thanks for using ${APP_NAME}. We will uninstall ${APP_NAME} and leave user settings intact. You can delete them from the $LOCALAPPDATA\${APP_NAME} folder. Hope to see you again."
  !insertmacro MUI_UNPAGE_WELCOME
  !insertmacro MUI_UNPAGE_CONFIRM
  !insertmacro MUI_UNPAGE_INSTFILES
  !insertmacro MUI_UNPAGE_FINISH

;--------------------------------
; languages
  !insertmacro MUI_LANGUAGE "English"

;--------------------------------
; installer

Section "MagellanMapper" SecMagMap
 
  SetOutPath "$INSTDIR"

  ; install MagellanMapper folder built by PyInstaller
  File /r "${BASEDIR}\${APP_NAME}\*.*"

  ; store installation folder in registry
  WriteRegStr HKCU "Software\${APP_NAME_VER}" "" $INSTDIR

  ; create uninstaller
  WriteUninstaller "$INSTDIR\Uninstall.exe"

  ; write apps add/remove descriptions
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
                 "DisplayName" "${APP_NAME}: a graphical imaging informatics suite for 3D reconstruction"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
                 "UninstallString" "$\"$INSTDIR\uninstall.exe$\""

  !insertmacro MUI_STARTMENU_WRITE_BEGIN Application
    
    ; create shortcuts
    ; need to set shell var context to appear in all users
    ;SetShellVarContext all
    CreateDirectory "$SMPROGRAMS\$StartMenuFolder"
    CreateShortCut "$SMPROGRAMS\$StartMenuFolder\${APP_NAME}.lnk" "$INSTDIR\${APP_NAME}.exe" "" "$INSTDIR\images\magmap.ico" 0 "SW_SHOWNORMAL" "" "${APP_NAME}: a graphical imaging informatics suite for 3D reconstruction"
    CreateShortCut "$SMPROGRAMS\$StartMenuFolder\${APP_NAME} Website.lnk" "https://github.com/sanderslab/magellanmapper" "" "" 0
    CreateShortCut "$SMPROGRAMS\$StartMenuFolder\Uninstall.lnk" "$INSTDIR\Uninstall.exe" ""

  
  !insertmacro MUI_STARTMENU_WRITE_END

SectionEnd

;--------------------------------
; languages

  ;Language strings
  LangString DESC_MAGMAP ${LANG_ENGLISH} "The ${APP_NAME} imaging informatics suite."

  ;Assign language strings to sections
  !insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
    !insertmacro MUI_DESCRIPTION_TEXT ${SecMagMap} $(DESC_MAGMAP)
  !insertmacro MUI_FUNCTION_DESCRIPTION_END


;--------------------------------
; uninstaller

Section "Uninstall"
  ;SetShellVarContext all
  
  ; remove entire installation directory, which was limited to a custom
  ; user-defined directory that contains the app name in the path
  RMDir /r "$INSTDIR"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"

  ; remove shortcuts, if any
  Delete "$SMPROGRAMS\${APP_NAME_VER}\*.*"

  !insertmacro MUI_STARTMENU_GETFOLDER Application $StartMenuFolder
    
  Delete "$SMPROGRAMS\$StartMenuFolder\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\$StartMenuFolder\${APP_NAME} Website.lnk"
  Delete "$SMPROGRAMS\$StartMenuFolder\Uninstall.lnk"
  RMDir "$SMPROGRAMS\$StartMenuFolder"

  ; remove installation directory path to avoid reinstalling to old version num
  DeleteRegKey HKCU "Software\${APP_NAME_VER}"
  DeleteRegKey /ifempty HKCU "Software\${APP_NAME_VER}"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"

SectionEnd
