#!/var/lib/sdsom/venv/bin/python
import sys
import urllib2
import json

from pprint import pprint


try:
    ip = sys.argv[1]
except IndexError:
    print '\nplease execute with ip argument\n'
    sys.exit()
    #ip = '10.10.5.151'

def request_ajax_data(url,body,referer=None,**headers):
    req = urllib2.Request(url)

    req.add_header('Content-Type', 'application/json')
    req.add_header('X-Requested-With','XMLHttpRequest')


    if referer:
        req.add_header('Referer',referer)

    if headers:
        for k in headers.keys():
            req.add_header(k,headers[k])

    postBody = json.dumps(body)

    response = urllib2.urlopen(req, postBody)
    
    data = response.read()

    return json.loads(data)



req = request_ajax_data("http://{}:6680/api/v1/sds/hardware/license/simple_query".format(ip), {})
info = req['data']

for each in info['license_data']:
    
    d={
        "active_time":99,
        "capacity":90,
        "machineid":each['machineid'],
        "sn":info['sn'],
        "version":"mos",
    }
    
    # license register
    req = request_ajax_data('http://10.10.2.61/license/register', d)
    
    data = {'hostid': each['hostid']}
    data.update(req)
    req = request_ajax_data("http://{}:6680/api/v1/sds/hardware/license/simple_update".format(ip), data)
    print data
    print req
