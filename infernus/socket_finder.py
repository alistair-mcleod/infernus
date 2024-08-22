import socket
from time import sleep

while True:
	try:
		#print("trying")
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.bind(('', 0))
		addr = s.getsockname()
		#print("found socket",addr[1])

		s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s2.bind(('', addr[1]+1))
		addr2 = s2.getsockname()
		#print("found socket",addr2[1])

		s3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s3.bind(('', addr2[1]+1))
		addr3 = s3.getsockname()
		#print("found socket",addr3[1])

		s4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s4.bind(('', addr3[1]+1))
		addr4 = s4.getsockname()
		#print("found socket",addr4[1])

		s5 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s5.bind(('', addr4[1]+1))
		addr5 = s5.getsockname()
		#print("found socket",addr5[1])

		s6 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s6.bind(('', addr5[1]+1))
		addr6 = s6.getsockname()
		#print("found socket",addr6[1])

		sleep(1)
		
		s.close()
		s2.close()
		s3.close()
		s4.close()
		s5.close()
		s6.close()


		print(addr[1])

		exit(0)

	except Exception as e:
		#print(e)
		sleep(1)
		#print("blah")
		continue