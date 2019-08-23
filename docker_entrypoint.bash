#!/bin/sh
#vncserver -xstartup /usr/bin/xterm -SecurityTypes None -localhost no --I-KNOW-THIS-IS-INSECURE -geometry 1024x720 -depth 32 :0
vncserver -xstartup /usr/bin/startlxde -SecurityTypes None -localhost no --I-KNOW-THIS-IS-INSECURE -geometry 1024x720 -depth 32 :0
pkill light-locker

DISPLAY=:0 xterm
