# ===== 构建阶段：编译 LIGGGHTS-PUBLIC（串行版，工况单核、跨工况并行）=====
FROM ubuntu:22.04 AS builder
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential gfortran ca-certificates     && rm -rf /var/lib/apt/lists/*
COPY liggghts-src /src/LIGGGHTS
WORKDIR /src/LIGGGHTS/src
RUN make -j$(nproc) stubs && make -j$(nproc) serial

# ===== 运行阶段：LIGGGHTS + Python MCP =====
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends     python3 python3-pip     && rm -rf /var/lib/apt/lists/*
COPY --from=builder /src/LIGGGHTS/src/lmp_serial /opt/liggghts/bin/liggghts
WORKDIR /app
COPY server.py requirements.txt /app/
COPY core /app/core
COPY templates /app/templates
COPY meshes /app/meshes
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
ENV MCP_TRANSPORT=streamable-http MCP_HOST=0.0.0.0 MCP_PORT=8001     WORK_DIR=/work HOST_WORK_DIR=/work LIGGGHTS_CMD=/opt/liggghts/bin/liggghts
EXPOSE 8001
CMD ["python3", "server.py"]
