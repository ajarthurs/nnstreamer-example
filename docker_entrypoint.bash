#!/bin/sh
vncserver -SecurityTypes None -localhost no --I-KNOW-THIS-IS-INSECURE -geometry 1024x720 -depth 32 :0

/bin/bash
