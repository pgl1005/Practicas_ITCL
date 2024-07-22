import subprocess

# Configuraciones
num_clients = 10
server_address = "127.0.0.1:8080"

processes = []

# Lanzar múltiples clientes
for i in range(num_clients):
    cmd = ["python", "metricasCliente.py", "--server_address", server_address, "--partition_id", str(i)]
    #cmd = ["python", "customClient.py"]
    p = subprocess.Popen(cmd)
    processes.append(p)

# Esperar a que todos los procesos terminen
for p in processes:
    p.wait()