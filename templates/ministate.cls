% ministate.cls - a compact LaTeX class for short reports.
%
% Vendored verbatim from Gregor Cerar's write-up, which publishes the class as an inline
% code block rather than a downloadable file:
%   https://gcerar.github.io/posts/2025-05-07-ministate-v3/
%   https://www.overleaf.com/latex/templates/minimalstatement/pzgpkvvrzyqj
% Licensed CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/).
%
% Used by `make report`, which renders the Markdown reports under datasets/*/reports/ to PDF
% through pandoc with -V documentclass=ministate. Keep this file in sync with the post rather
% than editing it here; local adjustments belong in templates/report-tables.tex.
%
\ProvidesClass{ministate}[2023/03/29 v3.0 Minimalist statement class]
\LoadClass[11pt,a4paper]{article}

\usepackage[utf8]{inputenc} % Retained for compatibility with pre-2018 LaTeX releases
\usepackage[T1]{fontenc}
\usepackage{lmodern}

\usepackage{microtype}

\usepackage[margin=0.8in]{geometry}
\usepackage{parskip}
\usepackage{fancyhdr}

\setlength{\headheight}{15.2pt}
\pagestyle{fancy}
\fancyhf{} % Clear all header and footer fields

%--------------------------------------------------%
%    Title, HeaderTitle, Author, HeaderAuthor,     %
%                 Custom Date                      %
%--------------------------------------------------%

\let\oldtitle\title
\let\oldauthor\author
\let\olddate\date

\def\@headertitle{}
\def\@headerauthor{}
\def\@headerdate{}

% Redefine the \title and \author commands
\renewcommand{\title}[1]{%
    \oldtitle{#1}%
    \ifx\@headertitle\@empty%
        \relax\def\@headertitle{#1}%
    \fi%
}
\renewcommand{\author}[1]{%
    \oldauthor{#1}%
    \ifx\@headerauthor\@empty%
        \relax\def\@headerauthor{#1}%
    \fi%
}

\renewcommand{\date}[1]{%
    \olddate{#1}%
    \ifx\@headerdate\@empty%
        \relax\def\@headerdate{#1}%
    \fi%
}

% Commands for explicitly setting the header title and header author
\newcommand{\headertitle}[1]{\def\@headertitle{#1}}
\newcommand{\headerauthor}[1]{\def\@headerauthor{#1}}

\fancypagestyle{ministate}{%
  \fancyhf{}% clear everything
  \fancyhead[L]{\textbf{\@headertitle}\ifx\@headerdate\@empty\else\ (\@headerdate)\fi}%
  \fancyhead[R]{\textbf{\@headerauthor}}%
  \fancyfoot[C]{\thepage}%
}

\pagestyle{ministate}

%--------------------------------------------------%
%                   Document Body                  %
%--------------------------------------------------%

% Optional header overrides:
% \title{Your Title Here}
% \author{Author Name}
% \headertitle{Short Header Title}
% \headerauthor{Short Header Author}
% \date{Custom Date or Empty String}

% Keep the ministate header on a page that calls \maketitle.
\usepackage{etoolbox}
\makeatletter
\patchcmd{\maketitle}%
  {\thispagestyle{plain}}%
  {\thispagestyle{ministate}}%
  {}{}
\makeatother

\makeatletter
\def\@maketitle{%
  \newpage
  \begin{center}%
    \let\footnote\thanks
    {\LARGE \@title\par}%
    \vskip 0.2em%
    {\large\begin{tabular}[t]{c}\@author\end{tabular}\par}%
    \vskip 0.2em%
    {\large \@date}\vskip 0.2em%
  \end{center}%
  \par
}%
\makeatother
