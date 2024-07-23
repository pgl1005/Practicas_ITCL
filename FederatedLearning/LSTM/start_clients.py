import subprocess

num_clients = 10
server_address = "127.0.0.1:8080"

processes = []

for i in range(num_clients):
    cmd = ["python", "paulaCliente.py", "--server_address", server_address, "--partition_id", str(i)]
    p = subprocess.Popen(cmd)
    processes.append(p)

for p in processes:
    p.wait() 
