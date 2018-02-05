#!/bin/bash

echo
echo "=====无输出为正常的项目：====="
echo
echo "2.1.6 root 用户环境变量的安全性"
echo $PATH | egrep '(^|:)(\.|:|$)'
find `echo $PATH | tr ':' ' '` -type d \( -perm -002 -o -perm -020 \) -ls
echo "...done";echo "--------"


echo "2.2.1 .netrc .rhosts文件"
find / -name .netrc
find / -name .rhosts
echo "...done";echo "--------"


echo "2.2.3 重要目录和文件的权限设置"
ret=`ls -l /etc/passwd /etc/shadow /etc/group /etc/security /etc/services /etc/rc*.d /etc /etc/rc.d/init.d /tmp /etc/inetd.conf | grep rwxrwxrwx`
if [ "$ret" ];then
    echo "存在777文件或目录"
fi
echo "...done";echo "--------"


echo "2.2.4 查找未授权的 SUID/SGID 文件"
for PART in `grep -v ^# /etc/fstab | awk '($6 != "0") {print $2 }'`; do
    find $PART \( -perm -04000 -o -perm -02000 \) -type f -xdev -print
done
echo "...done";echo "--------"


echo "2.2.5 检查任何人都有写权限的目录"
for PART in `awk '($3 == "ext2" || $3 == "ext3") \
    { print $2 }' /etc/fstab`; do
    find $PART -xdev -type d \( -perm -0002 -a ! -perm -1000 \) -print
done
echo "...done";echo "--------"


echo "2.2.6 查找任何人都有写权限的文件"
for PART in `grep -v ^# /etc/fstab | awk '($6 != "0") {print $2 }'`; do
    find $PART -xdev -type f \( -perm -0002 -a ! -perm -1000 \) -print
done
echo "...done";echo "--------"


echo "2.2.7 检查没有属主的文件"
for PART in `grep -v ^# /etc/fstab | awk '($6 != "0") {print $2 }'`; do
    find $PART -nouser -o -nogroup -print
done
echo "...done";echo "--------"


echo "2.2.8 检查异常隐含文件"
find / -name '*…*' 
find / -name '^G'
echo "...done";echo "--------"


echo "2.2.10 关闭telnet"
systemctl list-unit-files | grep "telnet"
echo "...done";echo "--------"


echo
echo "=====无输出为不正常的项目：====="
echo
echo "2.2.9 登录超时设置"
ret=`cat /etc/profile |grep -i TMOUT`
echo "...done";echo "--------"



echo
echo "=====需要查看输出的项目：====="
echo
echo "2.1.1 用户口令设置"
cat /etc/login.defs | egrep "PASS_MAX_DAYS|PASS_MIN_DAYS|PASS_WARN_AGE"
ret=`awk -F: '($2 == "") { print $1 }' /etc/shadow`
if [ "$ret" ];then
    echo "  存在空口令账户"
fi
echo "...done";echo "--------"

echo "2.1.2 用户口令强度要求"
echo -e "查询结果: \c"; cat /etc/pam.d/system-auth | grep pam_cracklib.so
echo "正确: password requisite pam_cracklib.so difok=3 minlen=8 ucredit=-1 lcredit=-1 dcredit=1"
echo "...done";echo "--------"


echo "2.1.3 用户锁定策略"
echo -e "查询结果: \c"; cat /etc/pam.d/system-auth | grep pam_tally.so
echo "正确: auth required pam_tally.so onerr=fail deny=10 unlock_time=300"
echo "...done";echo "--------"


# echo "2.1.4 root用户远程登录限制"
# /etc/ssh/sshd_config
# ret=`grep -v "^[[:space:]]*#" /etc/ssh/sshd_config|grep "PermitRootLogin no"`
# if [ ! "$ret" ];then
#     echo "  ssh 未禁止root登陆"
# fi
# ret=`grep -v "^[[:space:]]*#" /etc/ssh/sshd_config|egrep "^protocol\s*2|^Protocol\s*2"`
# if [ ! "$ret" ];then
#     echo "  ssh 协议版本未设置为 2"
# fi
# echo "...done";echo "--------"


echo "2.1.5 检查是否存在除 root 之外 UID 为 0 的用户"
ret=`awk -F: '($3 == 0) { print $1 }' /etc/passwd`
if [ "$ret" ];then
    echo "  $ret"
fi
echo "...done";echo "--------"


echo "2.1.7 口令重复次数限制"
echo -e "查询结果: \c"; cat /etc/pam.d/system-auth | grep "shadow nullok"
echo "正确: password sufficient pam_unix.so md5 shadow nullok try_first_pass use_authtok remember=5"
echo "...done";echo "--------"


echo "2.2.2 用户的 umask 安全配置"
grep -n 'umask' /etc/profile /etc/csh.login /etc/csh.cshrc /etc/bashrc  
echo "...done";echo "--------"


echo "2.2.12 关闭不必要的服务"
who -r ; chkconfig --list
echo "...done";echo "--------"


echo "4.1.1 系统core dump状态"
cat /etc/security/limits.conf | grep core
echo "...done";echo "--------"


echo "5.1.4 ssl版本"
openssl version
echo "...done";echo "--------"


