mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"ahabibi@ualberta.ca\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
