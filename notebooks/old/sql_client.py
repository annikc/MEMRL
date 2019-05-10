import pymysql
import uuid

server      = 'jenkins.c2g09w2ghsye.us-east-1.rds.amazonaws.com'
username    = 'jeremyforan'
password    = 'nafaweM3'
database    = 'shinjo'
port        = 3306

conn = pymysql.connect(host=server, user=username, passwd=password, db=database, port=port)
cur = conn.cursor()

unique_id = uuid.uuid4()