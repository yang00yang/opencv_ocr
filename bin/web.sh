Date=$(date +%Y%m%d%H%M)
  
if [ "$1" = "stop" ]; then
    echo "停止倾斜探测服务器"
    ps aux|grep python|grep server|awk '{print $2}'|xargs kill -9
    exit
fi

nohup \
gunicorn\
    web.server:app \
    --workers=1 \
    --worker-class=gevent \
    --bind=0.0.0.0:8083 \
    --timeout=300 \
    >> ./logs/opencvOCR_$Date.log 2>&1 &
