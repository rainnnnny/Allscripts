from sqlalchemy import create_engine, MetaData, Table, Integer, String, Column
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker

from sdsom.db.models import *

#import sys
#try:
#    newModel = sys.argv[1]
#except:
#    sys.exit("must transfer arg")

engine=create_engine('sqlite:///1', poolclass=QueuePool)

print 'engine tables:', engine.table_names()

# sm = sessionmaker(bind=engine, autocommit=True, expire_on_commit=False)
# print 'sessionmaker ', sm
# session = sm()
# print 'session:,', session, session.query(RoutineInspection).first().ftype

Base.metadata.create_all(engine)

print 'engine tables:', engine.table_names()


