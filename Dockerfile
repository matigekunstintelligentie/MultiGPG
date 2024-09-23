FROM continuumio/miniconda3:latest AS build

COPY . /pkg

RUN apt-get update --fix-missing \
  && apt-get install \
  && conda config --set verbosity 3 \
  && conda config --set ssl_verify false \
  && conda update --all
RUN conda env create -f /pkg/environment.yml -p /env --solver libmamba -v
RUN cd pkg && make main-build
RUN conda clean -afy
RUN conda run -p /env python -m pip install --no-deps /pkg
RUN conda config --set ssl_verify true

FROM gcr.io/distroless/base-debian10

COPY --from=build /env /env

