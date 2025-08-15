// 工具函数
const showToast = (message, type = 'info') => {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div class="toast-content">
            <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle'}"></i>
            <span>${message}</span>
        </div>
    `;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('show');
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }, 100);
};

document.addEventListener("DOMContentLoaded", function () {
    const img = document.getElementById("generated-img");
    const downloadBtn = document.getElementById("download-btn");
    const favoriteBtn = document.getElementById("favorite-btn");
    const history = document.getElementById("history");
    const loading = document.getElementById("loading");
    const imageBox = document.getElementById("imageBox");
    const styleSelect = document.getElementById("style-select");
    const uploadFile = document.getElementById("upload-file");
    const searchInput = document.getElementById("search-favorites");

    // 防抖函数
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // 生成图片
    window.generateImage = async function () {
        const style = styleSelect.value;
        try {
            loading.style.display = "flex";
            imageBox.style.display = "none";

            const response = await fetch(`/generate?style=${style}`);
            if (!response.ok) throw new Error('生成失败');
            
            const imageUrl = await response.text();
            img.src = imageUrl;
            
            await new Promise((resolve, reject) => {
                img.onload = resolve;
                img.onerror = reject;
            });

            loading.style.display = "none";
            imageBox.style.display = "block";
            favoriteBtn.style.display = "inline";
            downloadBtn.style.display = "inline";
            downloadBtn.href = imageUrl;

            // 添加收藏功能
            favoriteBtn.onclick = async () => {
                try {
                    const res = await fetch("/favorite", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ image_url: imageUrl })
                    });
                    const data = await res.json();
                    showToast(data.message, data.status === 'success' ? 'success' : 'error');
                } catch (err) {
                    showToast('收藏失败', 'error');
                }
            };

            // 添加到历史记录
            const historyItem = document.createElement("div");
            historyItem.className = 'gallery-item';
            historyItem.innerHTML = `<img src="${imageUrl}" alt="历史头像">`;
            history.insertBefore(historyItem, history.firstChild);

            showToast('生成成功！', 'success');
        } catch (err) {
            loading.style.display = "none";
            showToast('生成失败，请重试', 'error');
        }
    };

    // 上传图片
    window.uploadImage = async function () {
        const file = uploadFile.files[0];
        if (!file) {
            showToast('请选择要上传的图片', 'error');
            return;
        }

        // 验证文件类型
        if (!file.type.startsWith('image/')) {
            showToast('请选择图片文件', 'error');
            return;
        }

        // 验证文件大小（最大 5MB）
        if (file.size > 5 * 1024 * 1024) {
            showToast('图片大小不能超过5MB', 'error');
            return;
        }

        try {
            const formData = new FormData();
            formData.append('file', file);
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            showToast(data.message, data.status === 'success' ? 'success' : 'error');
            
            if (data.status === 'success') {
                uploadFile.value = ''; // 清空文件选择
            }
        } catch (err) {
            showToast('上传失败，请重试', 'error');
        }
    };

    // 搜索收藏
    window.searchFavorites = debounce(async function () {
        const keyword = searchInput.value.trim();
        try {
            const response = await fetch(`/search_favorites?keyword=${encodeURIComponent(keyword)}`);
            const data = await response.json();
            
            if (data.status === 'success') {
                const favorites = document.getElementById('favorites');
                favorites.innerHTML = '';
                
                if (data.images.length === 0) {
                    favorites.innerHTML = '<div class="no-results">没有找到匹配的收藏</div>';
                    return;
                }

                data.images.forEach(img => {
                    const div = document.createElement('div');
                    div.className = 'gallery-item';
                    div.innerHTML = `
                        <img src="/static/favorites/${data.userId}/${img}" alt="收藏头像">
                        <div class="item-overlay">
                            <button class="delete-btn" data-image-url="/static/favorites/${data.userId}/${img}">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    `;
                    
                    const deleteBtn = div.querySelector('.delete-btn');
                    deleteBtn.onclick = () => deleteFavorite(deleteBtn.dataset.imageUrl);
                    
                    favorites.appendChild(div);
                });
            } else {
                showToast(data.message, 'error');
            }
        } catch (err) {
            showToast('搜索失败，请重试', 'error');
        }
    }, 300);

    // 删除收藏
    async function deleteFavorite(imageUrl) {
        if (!confirm('确定要删除这张图片吗？')) return;
        
        try {
            const response = await fetch("/delete_favorite", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image_url: imageUrl })
            });
            
            const data = await response.json();
            showToast(data.message, data.status === 'success' ? 'success' : 'error');
            
            if (data.status === 'success') {
                const item = document.querySelector(`[data-image-url="${imageUrl}"]`).closest('.gallery-item');
                item.style.animation = 'fadeOut 0.3s ease';
                setTimeout(() => item.remove(), 300);
            }
        } catch (err) {
            showToast('删除失败，请重试', 'error');
        }
    }

    // 绑定删除按钮事件
    document.querySelectorAll(".delete-btn").forEach(btn => {
        btn.onclick = () => deleteFavorite(btn.dataset.imageUrl);
    });

    // 清空所有收藏
    const clearBtn = document.getElementById("clear-favorites");
    if (clearBtn) {
        clearBtn.onclick = async () => {
            if (!confirm('确定要清空所有收藏吗？此操作不可恢复！')) return;
            
            try {
                const response = await fetch("/clear_favorites", { method: "POST" });
                const data = await response.json();
                showToast(data.message, data.status === 'success' ? 'success' : 'error');
                
                if (data.status === 'success') {
                    const favorites = document.getElementById('favorites');
                    favorites.style.animation = 'fadeOut 0.3s ease';
                    setTimeout(() => {
                        favorites.innerHTML = '';
                        favorites.style.animation = '';
                    }, 300);
                }
            } catch (err) {
                showToast('操作失败，请重试', 'error');
            }
        };
    }

    // 添加搜索框回车事件
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            searchFavorites();
        }
    });

    // 添加文件拖放功能
    const uploadZone = document.querySelector('.upload-container');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
    });

    uploadZone.addEventListener('dragenter', () => uploadZone.classList.add('drag-over'));
    uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
    
    uploadZone.addEventListener('drop', (e) => {
        uploadZone.classList.remove('drag-over');
        uploadFile.files = e.dataTransfer.files;
        uploadImage();
    });
});