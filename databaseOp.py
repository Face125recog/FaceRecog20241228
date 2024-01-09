import pymysql
import datetime
class DbOperate():

    def __init__(self):
        self.conn=pymysql.connect(host='localhost',
                               user='root',
                               password='123456',
                               db='facereg',
                               port=3306,
                               charset='utf8'
                               )

#返回用户身份数据，格式[name, id]，这个得改一下
    def find_who(self, index):

        cur = self.conn.cursor()
        try:
            cur.execute('SELECT * FROM employee')
            result = cur.fetchall()
            who = result[index]   #这里index是0，列表从零开始，不用加1了
        except Exception as e:
            print(e)

        cur.close()
        return who[1], who[0]


#插入用户签到数据
    def register_date(self, id, name, tt):
        cur = self.conn.cursor()
        sql = "INSERT INTO record(id, name, date) VALUES(%d, '%s','%s');" % (id,
                name, tt)
        try:

            cur.execute(sql)
            self.conn.commit()
            print("you have successfully inserted data!")
        except Exception as e:
            print(e)
        cur.close()

#返回表table_name的个数
    def list_sum(self, table_name):
        cur = self.conn.cursor()
        sql = "SELECT COUNT(*) FROM " + table_name
        try:
            cur.execute(sql)
            self.conn.commit()
            result = cur.fetchall()
            print(result[0][0])
        except Exception as e:
            print(e)
        cur.close()
        return result[0][0]

#返回单个用户的全部签到日期，用于判定是否已经签到
    def return_date(self, name):
        cur = self.conn.cursor()
        sql = "SELECT date FROM record WHERE record.name='cxp';"
        try:
            self.cur.execute(sql)
            res = self.cur.fetchall()
            print(res[1][0])
            print(type(res[1][0]))
        except Exception as e:
            print(e)
        cur.close()

#删除雇员
    def delete_employee(self, index):
        cur = self.conn.cursor()
        sql = "DELETE FROM employee WHERE employee.ID=" + index
        try:
            cur.execute(sql)
            self.conn.commit()
        except Exception as e:
            print(e)
            cur.close()
            return 1

#添加雇员
    def add_employee(self, index, name):
        cur = self.conn.cursor()
        sql = "INSERT INTO employee(ID, name) VALUES(%d, '%s');" % (index,
                                                                    name)
        try:
            cur.execute(sql)
            self.conn.commit()
            print("you have successfully inserted data!")
        except Exception as e:
            print(e)
            cur.close()
            return 1

        cur.close()


    def close_connection(self):
        self.conn.close()



