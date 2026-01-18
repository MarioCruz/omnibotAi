#!/bin/bash
# Generate self-signed SSL certificates for HTTPS streaming

echo "Generating self-signed SSL certificates..."

openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes \
    -subj "/C=US/ST=Local/L=Local/O=PiCamera/OU=Stream/CN=localhost"

echo ""
echo "Certificates generated:"
echo "  - cert.pem (certificate)"
echo "  - key.pem (private key)"
echo ""
echo "Note: Browsers will show a security warning for self-signed certs."
echo "Click 'Advanced' and 'Proceed' to accept."
