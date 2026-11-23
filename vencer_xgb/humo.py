"""Prueba de humo: 1 evaluación TPE (diseño) + disjunta por algoritmo, para detectar errores y medir tiempos."""
import subprocess, sys, time, os
import modelos
nombres = sys.argv[1:] or modelos.ORDEN
for n in nombres:
    t0 = time.time()
    r = subprocess.run([sys.executable, "-W", "ignore", "ajuste_general.py", n, "1"], capture_output=True, text=True)
    lineas = (r.stdout + r.stderr).strip().splitlines()
    res = [l for l in lineas if l.startswith(f"[{n}]")]
    with open(os.path.join("resultados", "humo.txt"), "a", encoding="utf-8") as f:
        f.write(f"{n}: código {r.returncode}, {time.time() - t0:.0f}s | {res[-1] if res else ''}\n")
        if r.returncode:
            f.write("\n".join(lineas[-15:]) + "\n")
    print(n, r.returncode, f"{time.time() - t0:.0f}s", flush=True)
