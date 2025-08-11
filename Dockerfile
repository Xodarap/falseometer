FROM aisiuk/inspect-tool-support
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install pdfplumber PyPDF2 pdf2docx
CMD ["tail", "-f", "/dev/null"]