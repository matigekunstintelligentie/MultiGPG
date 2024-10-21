FROM continuumio/miniconda3:latest AS build

COPY environment.yml /pkg/environment.yml

RUN apt-get update --fix-missing \
  && apt-get install \
  && conda config --set verbosity 3 \
  && conda config --set ssl_verify false \
  && conda update --all
RUN conda env create -f /pkg/environment.yml -p /env --solver libmamba -v

COPY . /pkg

SHELL ["conda", "run", "-n", "gpg", "/bin/bash", "-c"]

RUN cd pkg && make
RUN conda clean -afy
RUN conda run -p /env python -m pip install --no-deps /pkg
RUN conda config --set ssl_verify true

FROM gcr.io/distroless/base-debian10

COPY --from=build /env /env

