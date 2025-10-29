git config --global --unset http.proxy
git config --global --unset https.proxy
git config --unset http.proxy
git config --unset https.proxy

# 2) Clear any shell env proxy vars for this session
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# 3) Verify nothing proxy-related remains
git config --global --list | grep -i proxy || echo "No global proxy"
git config --list | grep -i proxy || echo "No repo proxy"
env | grep -i proxy || echo "No env proxy"
