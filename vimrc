" essentials
syntax on
set number
set showcmd
set cursorline

" tabs = 4 spaces
set tabstop=4
set expandtab

" better searching
set incsearch                    
set hlsearch

" filetype specific indent
filetype indent on

" different colour for columns > 80
let &colorcolumn=join(range(81,999),",")
highlight ColorColumn ctermbg=235 guibg=#2c2d27
highlight Normal ctermfg=grey ctermbg=black

" auto-resize vim splits
autocmd VimResized * wincmd =
